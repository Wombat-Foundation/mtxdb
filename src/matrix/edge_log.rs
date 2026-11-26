//! A journaled write-ahead log of event edges.
//!
//! The CSR adjacency tails live outside the WAL, so a crash can lose edge rows
//! whose event bodies are durable. This log records each event's edges inside
//! the same journaled transaction machinery as everything else, so they become
//! durable with the same group commit and never need a tail fsync of their own.
//! The CSR becomes an index that is rebuilt from the log: rows are applied past
//! a watermark that is itself stored in the CSR (see
//! [`crate::csr_adjacency::CsrAdjacency::recover_edge_log`]).
//!
//! One log serves every scope. An entry carries its scope (the room's event
//! collection id) and the event with its `prev`, `auth` and relation edges.
//! Entries are numbered from 1 by a counter record, and are removed once the
//! rows they describe are durable in the CSR (see
//! [`crate::edge_log::EdgeLog::prune_through`]).

use std::sync::{Mutex, PoisonError};

use crate::database::Database;
use crate::layout::ShardType;
use crate::short_id::GraphReader;
use crate::storage::{NodeData, NodeId, StorageError};
use crate::template::{derive_collection_id, MEMBER_NAMESPACE_INTL};

const ENTRY_MAGIC: [u8; 4] = *b"EDLG";
const ENTRY_VERSION: u8 = 1;
const COUNTER_ID: NodeId = *b"MTXD-EDGELOG-CNT";
const PRUNED_ID: NodeId = *b"MTXD-EDGELOG-PRN";
const ENTRY_PREFIX: [u8; 8] = *b"EDGELOG1";
/// Reserved event id of a scope-purge boundary written before purges carried a
/// token. Recovery still honours it.
const PURGE_EVENT_ID: &str = "mtxdb-edge-log-purge-v1";
/// Prefix of a scope-purge boundary's reserved event id; the purge's token
/// follows as 16 hex digits.
const PURGE_EVENT_PREFIX: &str = "mtxdb-edge-log-purge-v2:";
/// Serialises [`EdgeLog::prune_through`] within the process.
static PRUNE_LOCK: Mutex<()> = Mutex::new(());
/// Entries removed per prune transaction, bounding one transaction's staging.
const PRUNE_BATCH: u64 = 512;

/// One logged event: the scope it was recorded in and its edges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoggedEvent {
    /// The room's event collection id, the scope the edges belong to.
    pub scope: [u8; 16],
    /// The event id.
    pub event_id: String,
    /// The event's `prev_events`.
    pub prev: Vec<String>,
    /// The event's `auth_events`.
    pub auth: Vec<String>,
    /// The event's relation, as `(target, rel_type)`, if it has one.
    pub relation: Option<(String, String)>,
}

/// Handle on the edge log's collection.
#[derive(Debug, Clone, Copy)]
pub struct EdgeLog {
    pool: ShardType,
    collection: [u8; 16],
}

impl Default for EdgeLog {
    fn default() -> Self {
        Self::new()
    }
}

/// A process-seeded random value that tells one purge from another.
fn fresh_token() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u64(NEXT.fetch_add(1, Ordering::Relaxed));
    hasher.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos()),
    );
    hasher.finish()
}

fn entry_id(seq: u64) -> NodeId {
    let mut id = [0_u8; 16];
    id[..8].copy_from_slice(&ENTRY_PREFIX);
    id[8..].copy_from_slice(&seq.to_be_bytes());
    id
}

fn put_str(out: &mut Vec<u8>, value: &str) -> Result<(), StorageError> {
    let len = u32::try_from(value.len())
        .map_err(|_| StorageError::Corrupt("edge log string too long".to_owned()))?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn put_list(out: &mut Vec<u8>, values: &[String]) -> Result<(), StorageError> {
    let len = u32::try_from(values.len())
        .map_err(|_| StorageError::Corrupt("edge log list too long".to_owned()))?;
    out.extend_from_slice(&len.to_be_bytes());
    for value in values {
        put_str(out, value)?;
    }
    Ok(())
}

fn encode(event: &LoggedEvent) -> Result<Vec<u8>, StorageError> {
    let mut out = Vec::with_capacity(64);
    out.extend_from_slice(&ENTRY_MAGIC);
    out.push(ENTRY_VERSION);
    out.extend_from_slice(&event.scope);
    put_str(&mut out, &event.event_id)?;
    put_list(&mut out, &event.prev)?;
    put_list(&mut out, &event.auth)?;
    match &event.relation {
        None => out.push(0),
        Some((target, rel_type)) => {
            out.push(1);
            put_str(&mut out, target)?;
            put_str(&mut out, rel_type)?;
        }
    }
    Ok(out)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], StorageError> {
        let end = self
            .at
            .checked_add(len)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| StorageError::Corrupt("truncated edge log entry".to_owned()))?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, StorageError> {
        let raw: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| StorageError::Corrupt("edge log length".to_owned()))?;
        Ok(u32::from_be_bytes(raw))
    }

    fn string(&mut self) -> Result<String, StorageError> {
        let len = self.u32()? as usize;
        String::from_utf8(self.take(len)?.to_vec())
            .map_err(|_| StorageError::Corrupt("edge log string is not utf-8".to_owned()))
    }

    fn list(&mut self) -> Result<Vec<String>, StorageError> {
        let count = self.u32()? as usize;
        (0..count).map(|_| self.string()).collect()
    }
}

fn decode(bytes: &[u8]) -> Result<LoggedEvent, StorageError> {
    let mut cursor = Cursor { bytes, at: 0 };
    if cursor.take(4)? != ENTRY_MAGIC || cursor.take(1)?[0] != ENTRY_VERSION {
        return Err(StorageError::Corrupt("edge log entry header".to_owned()));
    }
    let scope: [u8; 16] = cursor
        .take(16)?
        .try_into()
        .map_err(|_| StorageError::Corrupt("edge log scope".to_owned()))?;
    let event_id = cursor.string()?;
    let prev = cursor.list()?;
    let auth = cursor.list()?;
    let relation = match cursor.take(1)?[0] {
        0 => None,
        1 => Some((cursor.string()?, cursor.string()?)),
        _ => return Err(StorageError::Corrupt("edge log relation flag".to_owned())),
    };
    Ok(LoggedEvent {
        scope,
        event_id,
        prev,
        auth,
        relation,
    })
}

impl EdgeLog {
    /// The edge log, in the state pool beside the short-id maps.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pool: ShardType::State,
            collection: derive_collection_id(Some(MEMBER_NAMESPACE_INTL), b"matrix-edge-log:v1"),
        }
    }

    /// The sequence number of the newest entry ever written (0 for none). It
    /// only grows: pruning removes entries but never lowers it.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt counter.
    pub fn head<R: GraphReader + ?Sized>(&self, db: &R) -> Result<u64, StorageError> {
        let records = db.read_nodes(self.pool, &self.collection, &[COUNTER_ID])?;
        match records.into_iter().next().flatten() {
            None => Ok(0),
            Some(record) => {
                let raw: [u8; 8] = record
                    .bytes
                    .as_ref()
                    .try_into()
                    .map_err(|_| StorageError::Corrupt("edge log counter".to_owned()))?;
                u64::try_from(i64::from_be_bytes(raw))
                    .map_err(|_| StorageError::Corrupt("edge log counter range".to_owned()))
            }
        }
    }

    /// Log `events` in one journaled transaction and return the sequence number
    /// of the last one. The entries become durable with the next WAL group
    /// commit. Callers must serialise appends (the CSR store's write lock does),
    /// so sequence order is the order the rows are applied in.
    ///
    /// # Errors
    /// Returns an error if the transaction fails or the sequence overflows.
    pub fn append(&self, db: &Database, events: &[LoggedEvent]) -> Result<u64, StorageError> {
        self.append_inner(db, events, false)
    }

    /// Append a boundary that tells recovery to discard earlier entries for
    /// this scope before replaying later ones. Returns its sequence number and
    /// its token. Sequence numbers can be reused after a crash rolls the log
    /// back, so the token is what identifies one purge from another.
    pub(crate) fn append_scope_purge(
        &self,
        db: &Database,
        scope: [u8; 16],
    ) -> Result<(u64, u64), StorageError> {
        let token = fresh_token();
        let marker = LoggedEvent {
            scope,
            event_id: format!("{PURGE_EVENT_PREFIX}{token:016x}"),
            prev: Vec::new(),
            auth: Vec::new(),
            relation: None,
        };
        let seq = self.append_inner(db, &[marker], true)?;
        Ok((seq, token))
    }

    pub(crate) fn is_scope_purge(event: &LoggedEvent) -> bool {
        (event.event_id == PURGE_EVENT_ID || Self::purge_token(event).is_some())
            && event.prev.is_empty()
            && event.auth.is_empty()
            && event.relation.is_none()
    }

    /// The token of a scope-purge boundary, or `None` for an ordinary event or
    /// a boundary written before tokens existed.
    pub(crate) fn purge_token(event: &LoggedEvent) -> Option<u64> {
        let digits = event.event_id.strip_prefix(PURGE_EVENT_PREFIX)?;
        if digits.len() != 16 {
            return None;
        }
        u64::from_str_radix(digits, 16).ok()
    }

    fn append_inner(
        &self,
        db: &Database,
        events: &[LoggedEvent],
        allow_purge_marker: bool,
    ) -> Result<u64, StorageError> {
        if events.is_empty() {
            return self.head(db);
        }
        if !allow_purge_marker && events.iter().any(Self::is_scope_purge) {
            return Err(StorageError::Corrupt(
                "edge-log scope-purge event id is reserved".to_owned(),
            ));
        }
        let count = i64::try_from(events.len())
            .map_err(|_| StorageError::Corrupt("edge log batch too large".to_owned()))?;
        let txn = db.begin_transaction();
        let result = (|| -> Result<u64, StorageError> {
            let counters =
                txn.increment_counters(self.pool, self.collection, &[(COUNTER_ID, count)])?;
            let last = u64::try_from(counters[0])
                .map_err(|_| StorageError::Corrupt("edge log sequence overflow".to_owned()))?;
            let first = last
                .checked_sub(u64::try_from(events.len()).unwrap_or(u64::MAX))
                .and_then(|before| before.checked_add(1))
                .ok_or_else(|| StorageError::Corrupt("edge log sequence".to_owned()))?;
            for (offset, event) in events.iter().enumerate() {
                let seq = first
                    .checked_add(u64::try_from(offset).unwrap_or(u64::MAX))
                    .ok_or_else(|| StorageError::Corrupt("edge log sequence".to_owned()))?;
                let data = NodeData::new(bytes::Bytes::from(encode(event)?));
                txn.put(self.pool, self.collection, entry_id(seq), &data)
                    .map_err(StorageError::Io)?;
            }
            Ok(last)
        })();
        match result {
            Ok(last) => {
                txn.commit()?;
                Ok(last)
            }
            Err(error) => {
                let _ = txn.abort();
                Err(error)
            }
        }
    }

    /// The entries with `after < seq <= through`, in order. An entry that was
    /// pruned is skipped, so a caller that asks for a range it has already made
    /// durable in the CSR simply gets fewer entries.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt entry.
    pub fn read_range<R: GraphReader + ?Sized>(
        &self,
        db: &R,
        after: u64,
        through: u64,
    ) -> Result<Vec<(u64, LoggedEvent)>, StorageError> {
        let mut out = Vec::new();
        let mut next = after.saturating_add(1);
        while next <= through {
            let stop = through.min(next.saturating_add(PRUNE_BATCH - 1));
            let ids: Vec<NodeId> = (next..=stop).map(entry_id).collect();
            let records = db.read_nodes(self.pool, &self.collection, &ids)?;
            for (seq, record) in (next..=stop).zip(records) {
                if let Some(record) = record {
                    out.push((seq, decode(&record.bytes)?));
                }
            }
            if stop == u64::MAX {
                break;
            }
            next = stop.saturating_add(1);
        }
        Ok(out)
    }

    /// Remove one entry without touching the counters, as damage would.
    #[cfg(test)]
    pub(crate) fn drop_entry_for_test(&self, db: &Database, seq: u64) {
        let txn = db.begin_transaction();
        txn.remove_many(self.pool, self.collection, &[entry_id(seq)])
            .unwrap();
        txn.commit().unwrap();
    }

    /// The newest sequence number already pruned (0 for none).
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt counter.
    pub fn pruned<R: GraphReader + ?Sized>(&self, db: &R) -> Result<u64, StorageError> {
        let records = db.read_nodes(self.pool, &self.collection, &[PRUNED_ID])?;
        match records.into_iter().next().flatten() {
            None => Ok(0),
            Some(record) => {
                let raw: [u8; 8] = record
                    .bytes
                    .as_ref()
                    .try_into()
                    .map_err(|_| StorageError::Corrupt("edge log pruned counter".to_owned()))?;
                u64::try_from(i64::from_be_bytes(raw))
                    .map_err(|_| StorageError::Corrupt("edge log pruned range".to_owned()))
            }
        }
    }

    /// Raise the head and the pruned marker together to `target`, once every
    /// entry up to the current head has been pruned. Sequence numbers up to
    /// `target` are then never handed out again.
    ///
    /// Recovery uses this when a CSR scope records an applied purge at a
    /// sequence beyond the recovered head: the log was rolled back past that
    /// purge, and a reused number at or below it would be taken for an entry
    /// that precedes the purge. Does nothing if `target` is not above the head
    /// or entries are still unpruned.
    ///
    /// # Errors
    /// Returns an error if the transaction fails or a counter is corrupt.
    pub fn raise_head_to(&self, db: &Database, target: u64) -> Result<(), StorageError> {
        let _serial = PRUNE_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        let txn = db.begin_transaction();
        let (records, versions) =
            txn.get_with_record_versions(self.pool, &self.collection, &[COUNTER_ID, PRUNED_ID])?;
        let read = |record: Option<&NodeData>| -> Result<u64, StorageError> {
            let Some(record) = record else { return Ok(0) };
            let raw: [u8; 8] = record
                .bytes
                .as_ref()
                .try_into()
                .map_err(|_| StorageError::Corrupt("edge log counter".to_owned()))?;
            u64::try_from(i64::from_be_bytes(raw))
                .map_err(|_| StorageError::Corrupt("edge log counter range".to_owned()))
        };
        let head = read(records[0].as_ref())?;
        let pruned = read(records[1].as_ref())?;
        if target <= head || pruned != head {
            txn.abort()?;
            return Ok(());
        }
        let delta = i64::try_from(target.saturating_sub(head))
            .map_err(|_| StorageError::Corrupt("edge log raise range".to_owned()))?;
        for (id, version) in [COUNTER_ID, PRUNED_ID].into_iter().zip(&versions) {
            txn.expect_record_version(self.pool, self.collection, id, *version)
                .map_err(StorageError::Io)?;
        }
        txn.increment_counters(
            self.pool,
            self.collection,
            &[(COUNTER_ID, delta), (PRUNED_ID, delta)],
        )?;
        txn.commit()?;
        Ok(())
    }

    /// Remove every entry with `seq <= through`, once the rows they describe
    /// are durable in the CSR. A persisted marker remembers how far pruning
    /// got, so a later call removes only what is new instead of re-checking the
    /// whole history. The sequence counter is untouched. A `through` above the
    /// current head is lowered to it.
    ///
    /// # Errors
    /// Returns an error if a transaction fails.
    pub fn prune_through(&self, db: &Database, through: u64) -> Result<(), StorageError> {
        // The marker is advanced by a delta computed from what was read, so two
        // overlapping prunes could both add the same range. One at a time.
        let _serial = PRUNE_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        // Never advance the marker past the head: entries above it do not exist
        // yet, and a marker beyond them would hide the entries that later take
        // those numbers.
        let through = through.min(self.head(db)?);
        let mut done = self.pruned(db)?;
        while done < through {
            let txn = db.begin_transaction();
            // Re-read the marker inside this transaction and keep its version
            // as a commit precondition. The process mutex handles local
            // callers; the record-version check also prevents two processes
            // from applying deltas computed from stale marker values.
            let (records, versions) =
                txn.get_with_record_versions(self.pool, &self.collection, &[PRUNED_ID])?;
            let current = match records.into_iter().next().flatten() {
                None => 0,
                Some(record) => {
                    let raw: [u8; 8] =
                        record.bytes.as_ref().try_into().map_err(|_| {
                            StorageError::Corrupt("edge log pruned counter".to_owned())
                        })?;
                    u64::try_from(i64::from_be_bytes(raw))
                        .map_err(|_| StorageError::Corrupt("edge log pruned range".to_owned()))?
                }
            };
            if current >= through {
                txn.abort()?;
                return Ok(());
            }
            let first = current.saturating_add(1);
            let stop = through.min(first.saturating_add(PRUNE_BATCH - 1));
            let ids: Vec<NodeId> = (first..=stop).map(entry_id).collect();
            txn.expect_record_version(self.pool, self.collection, PRUNED_ID, versions[0])
                .map_err(StorageError::Io)?;
            txn.remove_many(self.pool, self.collection, &ids)
                .map_err(StorageError::Io)?;
            txn.increment_counters(
                self.pool,
                self.collection,
                &[(
                    PRUNED_ID,
                    i64::try_from(stop.saturating_sub(current))
                        .map_err(|_| StorageError::Corrupt("edge log prune range".to_owned()))?,
                )],
            )?;
            txn.commit()?;
            done = stop;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(room: u8, id: &str, prev: &[&str]) -> LoggedEvent {
        LoggedEvent {
            scope: [room; 16],
            event_id: id.to_owned(),
            prev: prev.iter().map(|p| (*p).to_owned()).collect(),
            auth: vec!["$create".to_owned()],
            relation: None,
        }
    }

    #[test]
    fn entries_round_trip_including_a_relation() {
        let mut with_relation = event(1, "$react", &["$msg"]);
        with_relation.relation = Some(("$msg".to_owned(), "m.annotation".to_owned()));
        for original in [
            event(1, "$a", &[]),
            event(2, "$b", &["$a", "$c"]),
            with_relation,
        ] {
            assert_eq!(decode(&encode(&original).unwrap()).unwrap(), original);
        }
    }

    #[test]
    fn a_damaged_entry_is_corrupt_not_a_panic() {
        let bytes = encode(&event(1, "$a", &["$p"])).unwrap();
        for cut in [0, 3, 5, 20, bytes.len() - 1] {
            assert!(decode(&bytes[..cut]).is_err(), "cut at {cut}");
        }
        let mut bad = bytes;
        bad[0] ^= 0xFF;
        assert!(decode(&bad).is_err());
    }

    #[test]
    fn append_numbers_entries_from_one_and_reads_them_back_in_order() {
        let root =
            crate::test_scratch::root().join(format!("mtxdb-edgelog-order-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        assert_eq!(log.head(&db).unwrap(), 0);

        assert_eq!(log.append(&db, &[event(1, "$a", &[])]).unwrap(), 1);
        assert_eq!(
            log.append(&db, &[event(1, "$b", &["$a"]), event(2, "$c", &[])])
                .unwrap(),
            3
        );
        assert_eq!(log.head(&db).unwrap(), 3);

        let entries = log.read_range(&db, 0, 3).unwrap();
        let ids: Vec<_> = entries
            .iter()
            .map(|(seq, e)| (*seq, e.event_id.as_str()))
            .collect();
        assert_eq!(ids, vec![(1, "$a"), (2, "$b"), (3, "$c")]);
        assert_eq!(
            log.read_range(&db, 1, 2).unwrap().len(),
            1,
            "after is exclusive"
        );
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn prune_removes_entries_but_the_head_never_goes_back() {
        let root =
            crate::test_scratch::root().join(format!("mtxdb-edgelog-prune-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        log.append(&db, &[event(1, "$a", &[]), event(1, "$b", &["$a"])])
            .unwrap();

        log.prune_through(&db, 1).unwrap();
        assert_eq!(log.pruned(&db).unwrap(), 1);
        log.prune_through(&db, 1).unwrap(); // already done: nothing to re-check

        let left: Vec<_> = log
            .read_range(&db, 0, 2)
            .unwrap()
            .into_iter()
            .map(|(seq, _)| seq)
            .collect();
        assert_eq!(left, vec![2]);
        assert_eq!(log.head(&db).unwrap(), 2);
        assert_eq!(log.append(&db, &[event(1, "$c", &["$b"])]).unwrap(), 3);
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }
}
