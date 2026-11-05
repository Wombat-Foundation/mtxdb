//! Explicit runtime configuration, including the opt-in environment reader.
//!
//! The library does **not** consult the environment on its own. A caller that
//! wants environment-driven defaults — the CLI, or an embedder that chooses to
//! opt in — calls [`RuntimeConfig::from_env`] once and passes the result to a
//! `*_with_config` constructor. Explicit API arguments therefore always win over
//! environment values, and tests never inherit an unrelated deployment's
//! settings.

use std::fmt;

use crate::index::IndexResidency;
use crate::journal::{MAX_RECLAIM_TRIGGER_LEN, MAX_TXN_STAGE_BYTES};
use crate::packfile::storage::DEFAULT_CACHE_CAPACITY;

/// Runtime knobs that are not part of a per-pool [`crate::PoolPolicy`].
///
/// The defaults preserve the library's historical behaviour:
/// - decoded-node cache: 100,000 entries per collection;
/// - transaction staging: the full 64 MiB engine ceiling;
/// - index residency: whatever the constructor or pool policy selected
///   (`Resident` in practice);
/// - read-path statistics: off;
/// - WAL reclaim trigger: the journal's built-in quarter of the segment cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeConfig {
    /// Decoded-node cache entries, applied independently to each collection.
    /// `0` disables the cache.
    pub cache_capacity: usize,
    /// Maximum charged payload one transaction may stage. Must be in
    /// `1..=MAX_TXN_STAGE_BYTES`.
    pub txn_stage_limit_bytes: usize,
    /// Optional override for every pool's index residency. `None` leaves the
    /// constructor or [`crate::PoolPolicy`] value in place.
    pub index_residency: Option<IndexResidency>,
    /// Enable the opt-in read-path counters and latency accounting.
    pub stats_enabled: bool,
    /// Segment length, in bytes, above which a sync forces a reclaiming
    /// checkpoint. `None` keeps the journal's built-in value. Must be in
    /// `1..=MAX_RECLAIM_TRIGGER_LEN` and below the live segment cap's
    /// emergency boundary.
    pub reclaim_trigger_bytes: Option<u64>,
    /// Optional cap, in bytes, on the history one replay lease may retain.
    /// `None` is unlimited (the default), matching `PostgreSQL`'s
    /// `max_slot_wal_keep_size`. On the next reclaim, a lease pinning more than
    /// this is revoked; its `advance` returns [`crate::ReplayLeaseRevoked`] and
    /// reads through its cursor fail closed.
    pub replay_lease_max_retained_bytes: Option<u64>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            cache_capacity: DEFAULT_CACHE_CAPACITY,
            txn_stage_limit_bytes: MAX_TXN_STAGE_BYTES,
            index_residency: None,
            stats_enabled: false,
            reclaim_trigger_bytes: None,
            replay_lease_max_retained_bytes: None,
        }
    }
}

impl RuntimeConfig {
    /// Read the supported `MTXDB_*` runtime variables.
    ///
    /// Unset variables keep their defaults. Invalid values are a hard error:
    /// silently falling back would make a constrained deployment look correct
    /// while running with the wrong budget.
    ///
    /// # Errors
    /// Returns [`ConfigError`] for an unparsable integer, an out-of-range
    /// transaction limit, an unknown residency, or an invalid boolean.
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_lookup(|name| match std::env::var(name) {
            Ok(value) => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(ConfigError::new(format!("{name} is not valid UTF-8")))
            }
        })
    }

    /// Shared parser, kept separate from `std::env` so it can be tested
    /// without mutating process-global state.
    fn from_lookup(
        lookup: impl Fn(&str) -> Result<Option<String>, ConfigError>,
    ) -> Result<Self, ConfigError> {
        let cache_capacity = match lookup("MTXDB_CACHE_CAPACITY")? {
            None => DEFAULT_CACHE_CAPACITY,
            Some(value) => value.trim().parse::<usize>().map_err(|_| {
                ConfigError::new(format!(
                    "MTXDB_CACHE_CAPACITY must be a non-negative integer, got {value:?}"
                ))
            })?,
        };

        let txn_stage_limit_bytes = match lookup("MTXDB_TXN_STAGE_LIMIT_BYTES")? {
            None => MAX_TXN_STAGE_BYTES,
            Some(value) => {
                let parsed = value.trim().parse::<usize>().map_err(|_| {
                    ConfigError::new(format!(
                        "MTXDB_TXN_STAGE_LIMIT_BYTES must be a positive integer, got {value:?}"
                    ))
                })?;
                if parsed == 0 || parsed > MAX_TXN_STAGE_BYTES {
                    return Err(ConfigError::new(format!(
                        "MTXDB_TXN_STAGE_LIMIT_BYTES must be in 1..={MAX_TXN_STAGE_BYTES}, got {parsed}"
                    )));
                }
                parsed
            }
        };

        let index_residency = match lookup("MTXDB_INDEX_RESIDENCY")? {
            None => None,
            Some(value) => Some(match value.trim().to_ascii_lowercase().as_str() {
                "resident" => IndexResidency::Resident,
                "paged" => IndexResidency::Paged,
                _ => {
                    return Err(ConfigError::new(format!(
                        "MTXDB_INDEX_RESIDENCY must be `resident` or `paged`, got {value:?}"
                    )));
                }
            }),
        };

        let stats_enabled = match lookup("MTXDB_STATS")? {
            None => false,
            Some(value) => parse_bool("MTXDB_STATS", &value)?,
        };

        let reclaim_trigger_bytes = match lookup("MTXDB_RECLAIM_TRIGGER_BYTES")? {
            None => None,
            Some(value) => {
                let parsed = value.trim().parse::<u64>().map_err(|_| {
                    ConfigError::new(format!(
                        "MTXDB_RECLAIM_TRIGGER_BYTES must be a positive integer, got {value:?}"
                    ))
                })?;
                check_reclaim_trigger(parsed)?;
                Some(parsed)
            }
        };

        let replay_lease_max_retained_bytes = match lookup("MTXDB_REPLAY_LEASE_MAX_RETAINED_BYTES")? {
            None => None,
            Some(value) => Some(value.trim().parse::<u64>().map_err(|_| {
                ConfigError::new(format!(
                    "MTXDB_REPLAY_LEASE_MAX_RETAINED_BYTES must be a non-negative integer, got {value:?}"
                ))
            })?),
        };

        Ok(Self {
            cache_capacity,
            txn_stage_limit_bytes,
            index_residency,
            stats_enabled,
            reclaim_trigger_bytes,
            replay_lease_max_retained_bytes,
        })
    }

    /// Validate values that a caller may have constructed directly.
    ///
    /// # Errors
    /// Returns [`ConfigError`] if the transaction staging limit is zero or
    /// above the engine ceiling.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.txn_stage_limit_bytes == 0 || self.txn_stage_limit_bytes > MAX_TXN_STAGE_BYTES {
            return Err(ConfigError::new(format!(
                "transaction staging limit must be in 1..={MAX_TXN_STAGE_BYTES}, got {}",
                self.txn_stage_limit_bytes
            )));
        }
        if let Some(bytes) = self.reclaim_trigger_bytes {
            check_reclaim_trigger(bytes)?;
        }
        Ok(())
    }

    /// Validate the configuration for a shared-WAL [`crate::Database`].
    ///
    /// Paged residency is rejected here, before the root lock or WAL is
    /// opened. The journaling path cannot advance durable coverage from a
    /// paged sync, so discovering that after modifying on-disk state is
    /// needlessly destructive.
    ///
    /// # Errors
    /// Returns [`ConfigError`] for a bad staging limit or Paged residency.
    pub fn validate_for_database(&self) -> Result<(), ConfigError> {
        self.validate()?;
        if self.index_residency == Some(IndexResidency::Paged) {
            return Err(ConfigError::new(
                "MTXDB_INDEX_RESIDENCY=paged cannot be used with Database/shared WAL; use `resident`",
            ));
        }
        Ok(())
    }
}

fn check_reclaim_trigger(bytes: u64) -> Result<(), ConfigError> {
    if bytes == 0 || bytes > MAX_RECLAIM_TRIGGER_LEN {
        return Err(ConfigError::new(format!(
            "reclaim trigger must be in 1..={MAX_RECLAIM_TRIGGER_LEN} bytes (below the emergency-zone boundary), got {bytes}"
        )));
    }
    Ok(())
}

fn parse_bool(name: &str, value: &str) -> Result<bool, ConfigError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Ok(true),
        "0" | "false" | "off" | "no" => Ok(false),
        _ => Err(ConfigError::new(format!(
            "{name} must be a boolean (1/0, true/false, on/off, yes/no), got {value:?}"
        ))),
    }
}

/// A rejected environment or runtime configuration value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    message: String,
}

impl ConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ConfigError {}

impl From<ConfigError> for std::io::Error {
    fn from(error: ConfigError) -> Self {
        Self::new(std::io::ErrorKind::InvalidInput, error.message)
    }
}

impl From<ConfigError> for crate::storage::StorageError {
    fn from(error: ConfigError) -> Self {
        Self::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            error.message,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(pairs: &[(&str, &str)]) -> Result<RuntimeConfig, ConfigError> {
        RuntimeConfig::from_lookup(|name| {
            Ok(pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned()))
        })
    }

    #[test]
    fn defaults_match_the_engine_defaults() {
        let parsed = config(&[]).unwrap();
        assert_eq!(parsed, RuntimeConfig::default());
        assert_eq!(parsed.cache_capacity, DEFAULT_CACHE_CAPACITY);
        assert_eq!(parsed.txn_stage_limit_bytes, MAX_TXN_STAGE_BYTES);
        assert_eq!(parsed.index_residency, None);
        assert!(!parsed.stats_enabled);
    }

    #[test]
    fn parses_the_supported_values() {
        let parsed = config(&[
            ("MTXDB_CACHE_CAPACITY", "0"),
            ("MTXDB_TXN_STAGE_LIMIT_BYTES", "1048576"),
            ("MTXDB_INDEX_RESIDENCY", "Paged"),
            ("MTXDB_STATS", "yes"),
        ])
        .unwrap();
        assert_eq!(parsed.cache_capacity, 0);
        assert_eq!(parsed.txn_stage_limit_bytes, 1 << 20);
        assert_eq!(parsed.index_residency, Some(IndexResidency::Paged));
        assert!(parsed.stats_enabled);
    }

    #[test]
    fn rejects_invalid_values_instead_of_falling_back() {
        assert!(config(&[("MTXDB_CACHE_CAPACITY", "lots")]).is_err());
        assert!(config(&[("MTXDB_TXN_STAGE_LIMIT_BYTES", "0")]).is_err());
        assert!(config(&[(
            "MTXDB_TXN_STAGE_LIMIT_BYTES",
            &(MAX_TXN_STAGE_BYTES + 1).to_string(),
        )])
        .is_err());
        assert!(config(&[("MTXDB_INDEX_RESIDENCY", "mmap")]).is_err());
        assert!(config(&[("MTXDB_STATS", "sometimes")]).is_err());
    }

    #[test]
    fn reclaim_trigger_parses_and_is_range_checked() {
        let parsed = config(&[("MTXDB_RECLAIM_TRIGGER_BYTES", " 4096 ")]).unwrap();
        assert_eq!(parsed.reclaim_trigger_bytes, Some(4096));
        assert_eq!(config(&[]).unwrap().reclaim_trigger_bytes, None);
        let boundary = MAX_RECLAIM_TRIGGER_LEN.to_string();
        assert_eq!(
            config(&[("MTXDB_RECLAIM_TRIGGER_BYTES", &boundary)])
                .unwrap()
                .reclaim_trigger_bytes,
            Some(MAX_RECLAIM_TRIGGER_LEN)
        );
        assert!(config(&[("MTXDB_RECLAIM_TRIGGER_BYTES", "0")]).is_err());
        assert!(config(&[("MTXDB_RECLAIM_TRIGGER_BYTES", "-1")]).is_err());
        assert!(config(&[("MTXDB_RECLAIM_TRIGGER_BYTES", "big")]).is_err());
        let over = (MAX_RECLAIM_TRIGGER_LEN + 1).to_string();
        assert!(config(&[("MTXDB_RECLAIM_TRIGGER_BYTES", &over)]).is_err());
        let direct = RuntimeConfig {
            reclaim_trigger_bytes: Some(0),
            ..RuntimeConfig::default()
        };
        assert!(direct.validate().is_err());
        let direct_over = RuntimeConfig {
            reclaim_trigger_bytes: Some(MAX_RECLAIM_TRIGGER_LEN.saturating_add(1)),
            ..RuntimeConfig::default()
        };
        assert!(direct_over.validate().is_err());
    }

    #[test]
    fn replay_lease_cap_parses_and_defaults_to_unlimited() {
        assert_eq!(config(&[]).unwrap().replay_lease_max_retained_bytes, None);
        let parsed = config(&[("MTXDB_REPLAY_LEASE_MAX_RETAINED_BYTES", " 4096 ")]).unwrap();
        assert_eq!(parsed.replay_lease_max_retained_bytes, Some(4096));
        let zero = config(&[("MTXDB_REPLAY_LEASE_MAX_RETAINED_BYTES", "0")]).unwrap();
        assert_eq!(zero.replay_lease_max_retained_bytes, Some(0));
        assert!(config(&[("MTXDB_REPLAY_LEASE_MAX_RETAINED_BYTES", "-1")]).is_err());
        assert!(config(&[("MTXDB_REPLAY_LEASE_MAX_RETAINED_BYTES", "big")]).is_err());
    }

    #[test]
    fn database_rejects_paged_before_opening() {
        let parsed = config(&[("MTXDB_INDEX_RESIDENCY", "paged")]).unwrap();
        assert!(parsed.validate_for_database().is_err());
    }
}
