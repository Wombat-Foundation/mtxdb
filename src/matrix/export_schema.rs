//! Wire schema of the `mtxdb-export/v1` interchange.
//!
//! A homeserver writes a room (or its signing keys) as JSONL; `mtxdb import`
//! reads it. Both sides must spell every record tag and field the same way and
//! agree on what each verdict code number means, so those live here once. The
//! module is pure: names, numbers and tag conversions, no JSON library and no
//! I/O. A `server_keys` record is one signing key of an origin, tagged by
//! [`key_kind::PROVISIONAL`] or [`key_kind::PERMANENT`], or the origin's
//! verbatim self-signed payload, tagged [`key_kind::NOTARY`].
//!
//! Each line is one JSON object tagged by [`field::T`]:
//!
//! 1. [`tag::HEADER`]: [`field::FORMAT`], [`field::SERVER`], [`field::ROOM_ID`].
//! 2. [`tag::EVENT`]: the event's identity, edges, flags, verdict codes and PDU.
//! 3. [`tag::END`]: [`field::EVENTS`], the number of event records. A file
//!    without it was truncated.
//!
//! A keys export is [`tag::SERVER_KEYS`] records only.

/// Format tag written to the header line. Bump it when the schema below
/// changes incompatibly; the importer rejects any other value.
pub const EXPORT_FORMAT: &str = "mtxdb-export/v1";

/// Record tags: the value of [`field::T`].
pub mod tag {
    /// First line of a room export.
    pub const HEADER: &str = "header";
    /// One event of a room export.
    pub const EVENT: &str = "event";
    /// Last line of a room export.
    pub const END: &str = "end";
    /// One signing-key record of a keys export.
    pub const SERVER_KEYS: &str = "server_keys";
}

/// The `kind` of a `server_keys` record.
pub mod key_kind {
    /// A key seen only through a notary so far.
    pub const PROVISIONAL: &str = "provisional";
    /// A key observed by a direct fetch from its origin.
    pub const PERMANENT: &str = "permanent";
    /// The origin's self-signed payload, kept verbatim.
    pub const NOTARY: &str = "notary";
}

/// Field names, by record.
pub mod field {
    /// The record tag, on every record.
    pub const T: &str = "t";

    /// `header`: the format tag, [`super::EXPORT_FORMAT`].
    pub const FORMAT: &str = "format";
    /// `header`: the exporting server's name.
    pub const SERVER: &str = "server";
    /// `header`: the room every event record belongs to.
    pub const ROOM_ID: &str = "room_id";

    /// `event`: the event's id.
    pub const EVENT_ID: &str = "event_id";
    /// `event`: ids of the event's `prev_events`.
    pub const PREV_EVENTS: &str = "prev_events";
    /// `event`: ids of the event's `auth_events`.
    pub const AUTH_EVENTS: &str = "auth_events";
    /// `event`: whether the event is an outlier.
    pub const OUTLIER: &str = "outlier";
    /// `event`: whether the event soft-failed.
    pub const SOFT_FAILED: &str = "soft_failed";
    /// `event`, optional: why it soft-failed, a [`super::SoftFailCode`] number.
    pub const SOFT_FAIL_CODE: &str = "soft_fail_code";
    /// `event`: whether the event was rejected.
    pub const REJECTED: &str = "rejected";
    /// `event`, optional: why it was rejected, a [`super::RejectionCode`] number.
    pub const REJECTION_CODE: &str = "rejection_code";
    /// `event`, optional: id of the event that redacted this one.
    pub const REDACTED_BY: &str = "redacted_by";
    /// `event`: the PDU as the source stored it.
    pub const PDU: &str = "pdu";

    /// `end`: how many `event` records the file holds.
    pub const EVENTS: &str = "events";

    /// `server_keys`: the server the keys belong to.
    pub const ORIGIN: &str = "origin";
    /// `server_keys`: the record kind, a [`super::key_kind`] tag.
    pub const KIND: &str = "kind";
    /// `server_keys` notary: base64 of the self-signed payload's exact bytes.
    pub const PAYLOAD: &str = "payload";
    /// `server_keys` key: the key's id, for example `ed25519:abc`.
    pub const KEY_ID: &str = "key_id";
    /// `server_keys` key: base64 of the public key bytes.
    pub const KEY: &str = "key";
    /// `server_keys` key: millisecond timestamp the key is valid until.
    pub const VALID_UNTIL_TS: &str = "valid_until_ts";
    /// `server_keys` key: who first supplied the key.
    pub const SOURCE: &str = "source";
    /// `server_keys` key: millisecond timestamp the key was last seen.
    /// `provisional` or `permanent`, optional: base64 SHA-256 of the key body.
    /// A key whose bytes were evicted carries this and no `key`.
    pub const DIGEST: &str = "digest";
    /// `provisional` or `permanent`, optional: when the receiver first bound
    /// the key id; absent is `ts_added_ms`.
    pub const FIRST_SEEN_MS: &str = "first_seen_ms";
    /// `provisional` or `permanent`, optional: when the key was last seen
    /// active; absent is `ts_added_ms`.
    pub const LAST_ACTIVE_MS: &str = "last_active_ms";
    /// `provisional` or `permanent`, optional: whether the receiver observed
    /// the key active; absent is false.
    pub const CORROBORATED: &str = "corroborated";
    /// `provisional` or `permanent`, optional: whether the key vanished from
    /// the origin's responses without an `expired_ts`; absent is false.
    pub const INFERRED_RETIRED: &str = "inferred_retired";
    /// `provisional`, optional: the last millisecond a direct fetch may still
    /// override the key; absent is `0`.
    pub const PROMOTE_BY_MS: &str = "promote_by_ms";
    /// `provisional` or `permanent`, optional: when the key was retired.
    pub const EXPIRED_TS: &str = "expired_ts";
    /// `provisional` or `permanent`: millisecond timestamp the key was last seen.
    pub const TS_ADDED_MS: &str = "ts_added_ms";
}

/// Why an event was rejected: the server refused it outright.
///
/// The numbers are part of the export format and of every stored verdict.
/// They are written out explicitly so renumbering a variant is a visible,
/// breaking edit; never reuse or reorder one. A number this build does not
/// know reads back as [`Self::Unknown`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum RejectionCode {
    /// The event's signatures did not verify.
    SignatureVerificationFailed = 0,
    /// The PDU is malformed.
    InvalidPduFormat = 1,
    /// An auth event it depends on was itself rejected.
    DependsOnRejectedAuthEvent = 2,
    /// Every prev event it builds on was rejected.
    PrevEventsRejected = 3,
    /// An auth event is missing and could not be fetched.
    MissingAuthEvent = 4,
    /// Two auth events claim the same (type, state key).
    DuplicateAuthEventKey = 5,
    /// The room's create event is missing.
    MissingCreateEvent = 6,
    /// The auth check against its resolved auth events failed.
    AuthCheckFailed = 7,
    /// A `/get_missing_events` response carried a structurally invalid prev
    /// event.
    StructurallyInvalidInGetMissingEvents = 8,
    /// A prev event stayed unknown after `/state_ids` recovery failed.
    PrevEventUnknownStateIdsFailed = 9,
    /// Every prev event was present but state recovery still failed.
    StateResolutionFailedWithPrevsPresent = 10,
    /// A reason this build cannot name.
    Unknown = 11,
}

impl RejectionCode {
    /// Every code, in number order.
    pub const ALL: [Self; 12] = [
        Self::SignatureVerificationFailed,
        Self::InvalidPduFormat,
        Self::DependsOnRejectedAuthEvent,
        Self::PrevEventsRejected,
        Self::MissingAuthEvent,
        Self::DuplicateAuthEventKey,
        Self::MissingCreateEvent,
        Self::AuthCheckFailed,
        Self::StructurallyInvalidInGetMissingEvents,
        Self::PrevEventUnknownStateIdsFailed,
        Self::StateResolutionFailedWithPrevsPresent,
        Self::Unknown,
    ];

    /// The code's number on the wire and on disk.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// The code for a number; any number this build does not define is
    /// [`Self::Unknown`].
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::SignatureVerificationFailed,
            1 => Self::InvalidPduFormat,
            2 => Self::DependsOnRejectedAuthEvent,
            3 => Self::PrevEventsRejected,
            4 => Self::MissingAuthEvent,
            5 => Self::DuplicateAuthEventKey,
            6 => Self::MissingCreateEvent,
            7 => Self::AuthCheckFailed,
            8 => Self::StructurallyInvalidInGetMissingEvents,
            9 => Self::PrevEventUnknownStateIdsFailed,
            10 => Self::StateResolutionFailedWithPrevsPresent,
            _ => Self::Unknown,
        }
    }

    /// The stable `snake_case` name stored in reason strings.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::SignatureVerificationFailed => "signature_verification_failed",
            Self::InvalidPduFormat => "invalid_pdu_format",
            Self::DependsOnRejectedAuthEvent => "depends_on_rejected_auth_event",
            Self::PrevEventsRejected => "prev_events_rejected",
            Self::MissingAuthEvent => "missing_auth_event",
            Self::DuplicateAuthEventKey => "duplicate_auth_event_key",
            Self::MissingCreateEvent => "missing_create_event",
            Self::AuthCheckFailed => "auth_check_failed",
            Self::StructurallyInvalidInGetMissingEvents => {
                "structurally_invalid_in_get_missing_events"
            }
            Self::PrevEventUnknownStateIdsFailed => "prev_event_unknown_state_ids_failed",
            Self::StateResolutionFailedWithPrevsPresent => {
                "state_resolution_failed_with_prevs_present"
            }
            Self::Unknown => "unknown",
        }
    }

    /// Read the code from a reason string: `tag` or `tag: detail`. `None` for
    /// a tag this build does not know (including `unknown` itself).
    #[must_use]
    pub fn parse(reason: &str) -> Option<Self> {
        let tag = reason.split(':').next().unwrap_or(reason).trim();
        Self::ALL
            .into_iter()
            .filter(|code| *code != Self::Unknown)
            .find(|code| code.tag() == tag)
    }
}

/// Why an event soft-failed: accepted into the DAG but not into current state.
///
/// Numbered explicitly for the same reason as [`RejectionCode`]; a number this
/// build does not know reads back as [`Self::Unknown`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum SoftFailCode {
    /// It failed auth against the room's current state.
    AuthCheckFailed = 0,
    /// It arrived soft-failed from an import.
    Imported = 1,
    /// An operator marked it.
    Manual = 2,
    /// A reason this build cannot name.
    Unknown = 3,
}

impl SoftFailCode {
    /// Every code, in number order.
    pub const ALL: [Self; 4] = [
        Self::AuthCheckFailed,
        Self::Imported,
        Self::Manual,
        Self::Unknown,
    ];

    /// The code's number on the wire and on disk.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// The code for a number; any number this build does not define is
    /// [`Self::Unknown`].
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::AuthCheckFailed,
            1 => Self::Imported,
            2 => Self::Manual,
            _ => Self::Unknown,
        }
    }

    /// The stable `snake_case` name stored in reason strings.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::AuthCheckFailed => "auth_check_failed",
            Self::Imported => "imported",
            Self::Manual => "manual",
            Self::Unknown => "unknown",
        }
    }

    /// Read the code from a reason string: `tag` or `tag: detail`. `None` for
    /// a tag this build does not know (including `unknown` itself).
    #[must_use]
    pub fn parse(reason: &str) -> Option<Self> {
        let tag = reason.split(':').next().unwrap_or(reason).trim();
        Self::ALL
            .into_iter()
            .filter(|code| *code != Self::Unknown)
            .find(|code| code.tag() == tag)
    }
}

#[cfg(test)]
#[path = "test_export_schema.rs"]
mod tests;
