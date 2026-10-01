//! Public façade over a room's auth closures.
//!
//! The façade wraps the lower-level closure primitives — the pure
//! [`AuthGraph`](crate::auth_closure::AuthGraph) computation and the persisted
//! generation layer — behind one operation-level error type,
//! [`RoomAuthError`](crate::room_auth::RoomAuthError), so callers do not have to interpret the storage layer's
//! taxonomy. The façade operations themselves land separately; this module
//! defines their shared error surface first.

use std::fmt;

use crate::storage::StorageError;

/// A room auth-closure operation failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum RoomAuthError {
    /// The room's short-id ordinal space is exhausted. Ids are never reused, so
    /// no further event can be assigned one.
    OrdinalExhausted,
    /// A requested event's `auth` chain references an event that was never
    /// recorded, so its closure cannot be computed.
    MissingParent {
        /// The absent event id.
        event_id: String,
    },
    /// A snapshot pinned a generation that a later publish has since retired.
    /// Take a fresh snapshot; retrying the same read cannot succeed.
    StaleGeneration {
        /// Generation the handle was pinned to.
        pinned: u64,
        /// Generation published now, or `None` if nothing is (for example the
        /// room was purged).
        current: Option<u64>,
    },
    /// A closure record belongs to a different room or scope than the one
    /// addressed.
    WrongDomain,
    /// A stored closure record is malformed or fails an integrity check.
    Corruption(String),
    /// The closure walk was incomplete: a reachable event has no recorded
    /// `auth` adjacency.
    IncompleteClosure {
        /// Event ids whose `auth` adjacency is absent.
        missing: Vec<String>,
    },
    /// The underlying storage layer failed.
    Storage(StorageError),
}

impl fmt::Display for RoomAuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OrdinalExhausted => write!(f, "room short-id space is exhausted"),
            Self::MissingParent { event_id } => {
                write!(f, "auth chain references unrecorded event {event_id}")
            }
            Self::StaleGeneration { pinned, current } => match current {
                Some(current) => write!(
                    f,
                    "pinned closure generation {pinned} is stale; current generation is {current}"
                ),
                None => write!(
                    f,
                    "pinned closure generation {pinned} is stale; no generation is published"
                ),
            },
            Self::WrongDomain => write!(f, "closure record belongs to a different domain"),
            Self::Corruption(message) => write!(f, "closure data is corrupt: {message}"),
            Self::IncompleteClosure { missing } => {
                write!(f, "auth closure is incomplete; missing {missing:?}")
            }
            Self::Storage(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for RoomAuthError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Storage(error) => Some(error),
            _ => None,
        }
    }
}

impl From<StorageError> for RoomAuthError {
    fn from(error: StorageError) -> Self {
        match error {
            StorageError::Corrupt(message) => Self::Corruption(message),
            StorageError::StaleGeneration {
                generation,
                current,
            } => Self::StaleGeneration {
                pinned: generation,
                current,
            },
            other => Self::Storage(other),
        }
    }
}
