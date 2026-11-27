//! Per-collection bucket indexes and their persistence formats.

pub mod bucket;
pub mod bucket_index;
pub mod checkpoint;
pub mod delta;
pub mod format;
pub mod redo;
pub mod versions;

pub use bucket_index::BucketIndex;

/// Per-database tuning for bucket-index construction.
#[derive(Debug, Clone, Copy)]
pub struct IndexConfig {
    /// Per-pool seed mixed into the fingerprint.
    pub seed: u64,
    /// Minimum initial capacity for a collection index.
    pub floor: u32,
    /// Maximum occupied percentage before a new distinct key requires growth.
    pub load_factor_percent: u8,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            floor: 64,
            load_factor_percent: 75,
        }
    }
}

impl IndexConfig {
    /// Validate the index configuration.
    ///
    /// # Errors
    /// Returns an error when the floor is zero or the load factor is outside
    /// `1..=90`.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.load_factor_percent == 0 || self.load_factor_percent > 90 {
            return Err("load_factor_percent must be in 1..=90");
        }
        if self.floor == 0 {
            return Err("floor must be > 0");
        }
        Ok(())
    }
}

/// Why an insertion could not be completed without growing the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertError {
    /// The index reached its configured load threshold or overflow limit.
    TableFull,
}

impl std::fmt::Display for InsertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TableFull => f.write_str("index table too full"),
        }
    }
}

impl std::error::Error for InsertError {}
