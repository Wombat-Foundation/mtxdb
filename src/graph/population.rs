//! A room's reconciliation population, read from the owner log.
//!
//! The owner log holds one entry per event the room's holder owns, each with
//! the 24-byte payload from [`encode_owner_payload`]. A [`PopulationSnapshot`]
//! pins the log at an `owner_seq_ceiling`, so an exchange keeps answering from
//! one population while writers append. Entries below the ceiling are
//! immutable and were committed with the counter that bounds them, so reading
//! them after the counter is safe.
//!
//! `manifest_version` names the compacted runs the snapshot also covers. No
//! runs exist yet, so it is always 0 and the population is the current log
//! generation alone.

use crate::database::Database;
use crate::short_id::ShortIdIndex;
use crate::storage::StorageError;
use rezzy_recon::triage::NodeSummary;
use rezzy_recon::{AlgebraicError, ElementHash, Population, SortedPopulation};

/// Length of an owner-log payload: `h64` then `h128`, both big-endian.
pub const OWNER_PAYLOAD_LEN: usize = 24;

/// The owner-log payload that adds `hash` to the population.
#[must_use]
pub fn encode_owner_payload(hash: ElementHash) -> [u8; OWNER_PAYLOAD_LEN] {
    let mut payload = [0_u8; OWNER_PAYLOAD_LEN];
    payload[..8].copy_from_slice(&hash.h64.to_be_bytes());
    payload[8..].copy_from_slice(&hash.h128.to_be_bytes());
    payload
}

fn decode_owner_payload(seq: u32, payload: &[u8]) -> Result<ElementHash, StorageError> {
    let corrupt =
        || StorageError::Corrupt(format!("owner-log entry {seq} payload is not 24 bytes"));
    let (h64, h128) = payload
        .split_first_chunk::<8>()
        .and_then(|(h64, rest)| Some((h64, <&[u8; 16]>::try_from(rest).ok()?)))
        .ok_or_else(corrupt)?;
    Ok(ElementHash {
        h64: u64::from_be_bytes(*h64),
        h128: u128::from_be_bytes(*h128),
    })
}

/// A population pinned at one point in the owner log.
#[derive(Debug, Clone)]
pub struct PopulationSnapshot {
    manifest_version: u32,
    owner_seq_ceiling: u32,
    population: SortedPopulation,
}

impl PopulationSnapshot {
    /// The compacted-run manifest this snapshot covers (0: no runs).
    #[must_use]
    pub fn manifest_version(&self) -> u32 {
        self.manifest_version
    }

    /// The owner-log sequence number the snapshot stops before: it holds
    /// every owner with `seq < owner_seq_ceiling` and none after.
    #[must_use]
    pub fn owner_seq_ceiling(&self) -> u32 {
        self.owner_seq_ceiling
    }

    /// Number of elements in the population.
    #[must_use]
    pub fn len(&self) -> usize {
        self.population.len()
    }

    /// Whether the population is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.population.is_empty()
    }
}

impl Population for PopulationSnapshot {
    fn node_summary(&self, node: (u8, u64)) -> Result<NodeSummary, AlgebraicError> {
        self.population.node_summary(node)
    }

    fn for_each_h64_in(&self, node: (u8, u64), f: &mut dyn FnMut(u64)) {
        self.population.for_each_h64_in(node, f);
    }

    fn candidates_into(&self, root: u64, out: &mut Vec<u128>) {
        self.population.candidates_into(root, out);
    }
}

impl ShortIdIndex {
    /// Pin the room's whole population at the current owner-log ceiling.
    ///
    /// Materializes the log once; every later read comes from memory.
    ///
    /// # Errors
    /// Returns an error if the log is unreadable or an entry's payload is not
    /// an [`encode_owner_payload`] value, or if the log has been compacted
    /// (runs are not readable yet).
    pub fn population_snapshot(&self, db: &Database) -> Result<PopulationSnapshot, StorageError> {
        let counters = self.counters(db)?;
        if counters.log_epoch_start_seq > 1 {
            return Err(StorageError::Internal(
                "owner log was compacted but no runs are readable".to_owned(),
            ));
        }
        let ceiling = counters.next_owner_seq;
        let entries = self.owner_log(db, counters.log_epoch, 1, ceiling)?;
        let elements = entries
            .iter()
            .map(|entry| decode_owner_payload(entry.seq, &entry.payload))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PopulationSnapshot {
            manifest_version: 0,
            owner_seq_ceiling: ceiling,
            population: SortedPopulation::new(elements),
        })
    }
}
