//! Rebuildable full-text search over event bodies, with no resident index.
//!
//! The primary event store stays authoritative. This module keeps a derived
//! *search pack*: one collection of densely packed, pre-extracted records that
//! exist only to be scanned. Nothing is retained in memory between queries —
//! a query walks the pack once, applying a substring search to the body and
//! cheap equality/range tests to the fixed header.
//!
//! Extracting fields at write time is what makes this cheap: a query never
//! parses event JSON, it only decodes a fixed-width header and scans bytes.
//! Because the searched text *is* the record payload, every query reads the
//! whole pack, so cost is linear in pack size. See
//! `docs/docs/2026-10-01-search-pack-design.md` for the measurements and the
//! documented escalation path (time-bucketed collections, then a per-block
//! sidecar).
//!
//! The pack is a real collection, so the engine's own machinery supplies
//! snapshot consistency and last-write-wins for free: re-indexing an edited
//! event overwrites the single record derived from its event id, and a scan is
//! taken against captured pack lengths.

use std::cmp::Ordering as CmpOrdering;
use std::collections::BinaryHeap;

use memchr::memmem;

use crate::storage::{DigestAlgorithm, NodeData, NodeId, StorageEngine, StorageError};
use crate::template::{
    derive_collection_id, record_logical_id, CollectionMetadata, FrameIdPolicy, PayloadPolicy,
    RecordIdentityRule, COLLECTION_METADATA_RECORD_ID, MEMBER_NAMESPACE_INTL,
};
use crate::PackfileStorage;

/// Version byte on every search record, so a pack written by a different
/// encoding is reported rather than silently mis-decoded.
///
/// `2` is the first encoding with a flags byte. Version `1` described the
/// header before redaction existed and would misparse these records by reading
/// the flags as the first byte of the timestamp, so the bump is what makes the
/// two distinguishable. No version-1 pack was ever written to disk, so there is
/// nothing to migrate; the cost of bumping now is zero and the cost of bumping
/// after a release is a compatibility rule.
pub(crate) const RECORD_VERSION: u8 = 1;

/// Body text was stripped by [`SearchIndexes::redact`]. Header fields are kept,
/// so the event stays findable by room, sender, type, and timestamp while its
/// body matches no text query.
pub(crate) const FLAG_REDACTED: u8 = 0x01;

/// Every searchable field was cleared by [`SearchIndexes::remove`].
///
/// Stronger than [`FLAG_REDACTED`]: a redacted event keeps a findable header,
/// a removed one matches nothing at all. The record still exists — mtxdb never
/// deletes — but it carries no room, sender, type, timestamp, event id or body,
/// and the scan skips it outright rather than trusting the empty fields.
pub(crate) const FLAG_REMOVED: u8 = 0x02;

/// Flag bits this build understands. Every other bit is rejected as corruption,
/// so a record carrying a flag we cannot honour fails loudly instead of being
/// read as an ordinary one.
///
/// Unknown bits fail closed on purpose: reading a record whose writer meant it
/// to be invisible would silently expose it, so the decoder reports the record
/// rather than approximating it.
pub(crate) const KNOWN_FLAGS: u8 = FLAG_REDACTED | FLAG_REMOVED;

/// Length of a record's fixed header: version, flags, timestamp, and four field
/// lengths. The timestamp sits early so a time-bounded query can reject a
/// record without touching any variable-length field.
const HEADER_LEN: usize = 1 + 1 + 8 + 4 * 2;

/// Domain separator mixed into the record id derivation, so a search record id
/// can never coincide with an id some other scheme derives from the same event
/// id. Record ids must never be a constant prefix: that would cluster every
/// record into one index bucket.
const ID_DOMAIN: &[u8] = b"mtxdb:search:v1\0";

/// A searchable event supplied by the ingestion layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchDocument {
    /// Matrix event id. Doubles as the record's identity, so re-indexing the
    /// same event replaces rather than duplicates.
    pub event_id: String,
    /// Room the event belongs to.
    pub room_id: String,
    /// Sender's user id.
    pub sender: String,
    /// Event type, e.g. `m.room.message`.
    pub event_type: String,
    /// Origin-server timestamp, in milliseconds.
    pub timestamp: u64,
    /// Decrypted body text. `None` for encrypted events, which are still
    /// indexed by their header fields but contribute no searchable text.
    pub body: Option<String>,
}

/// Search predicates. Terms are AND-ed substrings matched against the
/// lowercased body; an empty term list matches every indexed event.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SearchQuery {
    /// Substrings that must all appear in the body. Matched
    /// case-insensitively (ASCII).
    pub terms: Vec<String>,
    /// Exact room id match.
    pub room_id: Option<String>,
    /// Exact sender match.
    pub sender: Option<String>,
    /// Exact event type match.
    pub event_type: Option<String>,
    /// Inclusive `(start, end)` origin-server timestamp range.
    pub time_range: Option<(u64, u64)>,
    /// Maximum results, newest first. `0` means no limit.
    pub limit: usize,
}

/// The search pack: a single derived collection scanned to answer queries.
pub struct SearchIndexes<'a> {
    engine: &'a PackfileStorage,
    name: String,
    collection_id: [u8; 16],
}

impl<'a> SearchIndexes<'a> {
    /// Open (or create, on first write) the search pack for `namespace`.
    ///
    /// Opening is cheap and does not touch the filesystem: the collection is
    /// only established when the first document is indexed.
    #[must_use]
    pub fn open(engine: &'a PackfileStorage, namespace: &str) -> Self {
        let name = format!("{namespace}:searchpack");
        Self {
            collection_id: derive_collection_id(Some(MEMBER_NAMESPACE_INTL), name.as_bytes()),
            engine,
            name,
        }
    }

    /// The collection id this search pack is stored under.
    #[must_use]
    pub fn collection_id(&self) -> [u8; 16] {
        self.collection_id
    }

    /// Number of records currently indexed, including the collection's genesis
    /// metadata record, or `None` if the pack has never been written.
    ///
    /// # Errors
    /// Returns [`StorageError::Io`] on I/O failure.
    pub fn len(&self) -> Result<Option<usize>, StorageError> {
        self.engine.collection_len(&self.collection_id)
    }

    /// Add one document, replacing any existing record for the same event id.
    ///
    /// Idempotent: re-indexing an unchanged document appends nothing. A
    /// document whose fields changed since it was last indexed replaces the
    /// stored record in place.
    ///
    /// Redaction and removal are both sticky, so this silently does nothing for
    /// an event that has been redacted or removed: an ordinary re-ingest cannot
    /// put a taken-down body back, nor resurrect an event that was deleted.
    /// Clearing a redaction is an explicit operation, or a rebuild from a
    /// primary store that carries the redaction itself. That is a deliberate
    /// asymmetry — an importer that could quietly undo a takedown would be a
    /// standing bypass of it.
    ///
    /// The existence check costs one read per event. That is the right trade
    /// while nothing calls this in a hot loop, and it should be measured when
    /// ingest is wired up; dropping it would trade a takedown guarantee for a
    /// read.
    ///
    /// # Errors
    /// Returns [`StorageError::Io`] on read or write failure, or
    /// [`StorageError::Corrupt`] if a field is too long to encode, or if the
    /// record already stored for this event cannot be decoded.
    pub fn index(&self, document: &SearchDocument) -> Result<(), StorageError> {
        let node_id = record_id(&document.event_id);
        if self.is_sticky(&node_id)? {
            return Ok(());
        }
        let data = NodeData::new(bytes::Bytes::from(encode(document)?));
        self.engine.create_or_upsert_established_validated(
            &self.collection_id,
            &self.metadata(),
            &node_id,
            &data,
            &mut |_existing| Ok(()),
        )
    }

    /// Whether the record stored under `node_id` is one [`Self::index`] must
    /// leave alone.
    ///
    /// Both sticky states qualify: [`FLAG_REDACTED`] (body taken down, header
    /// kept) and [`FLAG_REMOVED`] (the record is invisible). An event that was
    /// never indexed is neither, and stays writable.
    fn is_sticky(&self, node_id: &NodeId) -> Result<bool, StorageError> {
        let Some(existing) = self.engine.get(&self.collection_id, node_id)? else {
            return Ok(false);
        };
        Ok(Record::decode(&existing.bytes)?.flags & (FLAG_REDACTED | FLAG_REMOVED) != 0)
    }

    /// Add several documents.
    ///
    /// Each document goes through the same upsert as [`Self::index`], so this
    /// is correct for re-ingest but does not coalesce writes into one batch.
    /// Returns the number of documents submitted. Documents suppressed by an
    /// earlier redaction or removal still count: they were submitted, not
    /// written.
    ///
    /// # Errors
    /// Propagates the first failure from [`Self::index`]; documents already
    /// written in this call stay written.
    pub fn index_many(&self, documents: &[SearchDocument]) -> Result<usize, StorageError> {
        for document in documents {
            self.index(document)?;
        }
        Ok(documents.len())
    }

    /// Strip an event's searchable text, keeping its header fields.
    ///
    /// mtxdb never deletes or mutates a record — a redaction is a new version
    /// of the same record, carrying `FLAG_REDACTED` and an empty body. The
    /// event remains findable by room, sender, type, and timestamp while its
    /// body stops matching any text query.
    ///
    /// The flag is sticky: [`Self::index`] will not overwrite it, so a later
    /// re-ingest of the same event cannot restore the text. Clearing a
    /// redaction is an explicit operation, or a rebuild from a primary store
    /// that records the redaction itself.
    ///
    /// Redacting a record that was already [`Self::remove`]d reports `true` and
    /// writes nothing: a removal is strictly stronger than a redaction, and
    /// rewriting it here would put the header fields back.
    ///
    /// This does **not** erase the plaintext from disk. The pack is append-only,
    /// so the pre-redaction frame survives until a repack or a rebuild drops
    /// superseded versions. See the design doc for what that does and does not
    /// guarantee.
    ///
    /// Idempotent: redacting an already-redacted record writes nothing.
    ///
    /// # Concurrency
    /// This is a read-then-write and is correct only under a single-writer
    /// assumption — one ingest or redaction path at a time per pack. A
    /// concurrent [`Self::index`] for the same event can interleave between the
    /// read and the write and win, because the engine exposes no
    /// compare-and-swap over a record's current version. Serializing writers is
    /// the caller's job.
    ///
    /// Returns `false` if the event was not in the pack.
    ///
    /// # Errors
    /// Returns [`StorageError::Io`] on read or write failure, or
    /// [`StorageError::Corrupt`] if the stored record cannot be decoded.
    pub fn redact(&self, event_id: &str) -> Result<bool, StorageError> {
        let node_id = record_id(event_id);
        let Some(existing) = self.engine.get(&self.collection_id, &node_id)? else {
            return Ok(false);
        };
        let record = Record::decode(&existing.bytes)?;
        if record.flags & (FLAG_REDACTED | FLAG_REMOVED) != 0 {
            // Already terminal. Redacting a redacted record is a no-op;
            // redacting a *removed* one must not write the header fields back.
            return Ok(true);
        }
        // Same encoder as an ordinary write, so the redacted record has exactly
        // the shape a redaction is supposed to produce.
        let replacement = encode_fields(
            FLAG_REDACTED,
            record.timestamp,
            record.room_id,
            record.sender,
            record.event_type,
            record.event_id.as_bytes(),
            &[],
        )?;
        self.engine.create_or_upsert_established_validated(
            &self.collection_id,
            &self.metadata(),
            &node_id,
            &NodeData::new(bytes::Bytes::from(replacement)),
            &mut |_existing| Ok(()),
        )?;
        Ok(true)
    }

    /// Drop an event from the pack: clear every searchable field and mark the
    /// record `FLAG_REMOVED`.
    ///
    /// [`Self::redact`] keeps the header so the event stays findable by room,
    /// sender, type and timestamp. `remove` is the stronger operation, for when
    /// those fields are themselves the sensitive part: the new version of the
    /// record carries no room, sender, type, timestamp or event id, and the
    /// flag makes the scan skip it outright instead of trusting the empty
    /// fields — an empty query must not surface it as a blank hit either.
    ///
    /// Like a redaction this is a new *version* of the record, not a tombstone,
    /// so mtxdb's write-once rule still holds. It also inherits redaction's
    /// limits: the pack is append-only, so the pre-removal frame survives until
    /// a repack or a rebuild drops superseded versions. This removes the event
    /// from search results; it does not erase the bytes from disk.
    ///
    /// Sticky: [`Self::index`] will not write over a removed record, so a later
    /// re-ingest cannot bring the event back. [`Self::redact`] on a removed
    /// record reports `true` and writes nothing, so it cannot downgrade a
    /// removal into a findable header.
    ///
    /// Idempotent: removing an already-removed record writes nothing.
    ///
    /// # Concurrency
    /// The same single-writer caveat as [`Self::redact`]: this is a
    /// read-then-write, and a concurrent [`Self::index`] for the same event can
    /// interleave between the read and the write and win. Serializing writers
    /// is the caller's job.
    ///
    /// Returns `false` if the event was not in the pack.
    ///
    /// # Errors
    /// Returns [`StorageError::Io`] on read or write failure, or
    /// [`StorageError::Corrupt`] if the stored record cannot be decoded.
    pub fn remove(&self, event_id: &str) -> Result<bool, StorageError> {
        let node_id = record_id(event_id);
        let Some(existing) = self.engine.get(&self.collection_id, &node_id)? else {
            return Ok(false);
        };
        let record = Record::decode(&existing.bytes)?;
        if record.flags & FLAG_REMOVED != 0 {
            return Ok(true);
        }
        // Every declared field zero-length and the body empty: the payload
        // holds nothing but the version, the flag and four zero lengths.
        let replacement = encode_fields(FLAG_REMOVED, 0, &[], &[], &[], &[], &[])?;
        self.engine.create_or_upsert_established_validated(
            &self.collection_id,
            &self.metadata(),
            &node_id,
            &NodeData::new(bytes::Bytes::from(replacement)),
            &mut |_existing| Ok(()),
        )?;
        Ok(true)
    }

    /// Drop the whole search pack. The primary event store is untouched, so
    /// this only discards derived data.
    ///
    /// # Errors
    /// Returns [`StorageError::Io`] on I/O failure.
    pub fn clear(&self) -> Result<(), StorageError> {
        self.engine.delete_collection(&self.collection_id)
    }

    /// Return the event ids of every indexed document satisfying `query`,
    /// newest first.
    ///
    /// Results are ordered by descending timestamp, with the event id as a
    /// tie-break so the order is total and therefore deterministic. The scan
    /// itself yields records in unspecified order, so this ordering is imposed
    /// here.
    ///
    /// When `query.limit` is non-zero the result set is bounded: matches are
    /// accumulated in a heap of at most `limit` entries, so memory does not
    /// scale with the number of hits. Only the newest `limit` survive.
    ///
    /// Cost is one pass over the pack, proportional to its size.
    ///
    /// # Errors
    /// Returns [`StorageError::Io`] on read failure, or [`StorageError::Corrupt`]
    /// if a stored record cannot be decoded.
    pub fn search(&self, query: &SearchQuery) -> Result<Vec<String>, StorageError> {
        if self.engine.collection_len(&self.collection_id)?.is_none() {
            return Ok(Vec::new());
        }

        // Terms are lowercased once, and each `Finder` is built once and reused
        // for every record — the scan is the hot loop, so per-record needle
        // construction would dominate it.
        let lowered: Vec<String> = query
            .terms
            .iter()
            .map(|term| term.to_ascii_lowercase())
            .filter(|term| !term.is_empty())
            .collect();
        let matcher = Matcher {
            finders: lowered
                .iter()
                .map(|term| memmem::Finder::new(term.as_bytes()))
                .collect(),
            room_id: query.room_id.as_deref().map(str::as_bytes),
            sender: query.sender.as_deref().map(str::as_bytes),
            event_type: query.event_type.as_deref().map(str::as_bytes),
            time_range: query.time_range,
        };

        let bounded = query.limit != 0;
        let scan = self.engine.scan_collection(&self.collection_id)?;
        // A max-heap of hits, ordered so that the *worst* surviving hit is the
        // maximum; that is the one dropped once the heap is full.
        let mut top: BinaryHeap<Hit> = BinaryHeap::new();
        for entry in scan {
            let (node_id, data) = entry?;
            // The collection's genesis metadata record lives in the same
            // collection but is not a search record.
            if node_id == COLLECTION_METADATA_RECORD_ID {
                continue;
            }
            let record = Record::decode(&data.bytes)?;
            if !matcher.matches(&record) {
                continue;
            }
            top.push(Hit {
                timestamp: record.timestamp,
                event_id: record.event_id.to_owned(),
            });
            if bounded && top.len() > query.limit {
                top.pop();
            }
        }

        let mut hits = top.into_vec();
        hits.sort_unstable_by(Hit::newest_first);
        Ok(hits.into_iter().map(|hit| hit.event_id).collect())
    }

    fn metadata(&self) -> CollectionMetadata {
        CollectionMetadata {
            member_namespace: Some(MEMBER_NAMESPACE_INTL),
            collection_canonical_id: self.name.as_bytes().to_vec(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::Key,
                digest_algorithm: DigestAlgorithm::Blake3,
            },
            payload: PayloadPolicy::Source,
            extension: None,
            role: Some("system_search".to_owned()),
            schema: None,
        }
    }
}

/// One match, ordered for two different purposes.
///
/// [`Ord`] is the *reverse* of the displayed order, so that the maximum of a
/// `BinaryHeap<Hit>` is the hit that would appear last — the one to discard
/// when the heap is full. [`Hit::newest_first`] is the display order.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Hit {
    timestamp: u64,
    event_id: String,
}

impl Hit {
    /// Newest first, event id ascending to break timestamp ties.
    fn newest_first(&self, other: &Self) -> CmpOrdering {
        other
            .timestamp
            .cmp(&self.timestamp)
            .then_with(|| self.event_id.cmp(&other.event_id))
    }
}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Deliberately the same comparator as `newest_first`. It reads
        // backwards because a max-heap and an ascending sort disagree about
        // what "greatest" means: a sort places the greatest element last,
        // which is the oldest hit, and a heap pops the greatest element first,
        // which should also be the oldest hit. Sharing one comparator keeps the
        // discarded element and the last-placed element the same by
        // construction rather than by two definitions agreeing.
        self.newest_first(other)
    }
}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

/// A decoded search record, borrowing its fields out of the stored payload.
struct Record<'a> {
    flags: u8,
    timestamp: u64,
    room_id: &'a [u8],
    sender: &'a [u8],
    event_type: &'a [u8],
    event_id: &'a str,
    body: &'a [u8],
}

impl<'a> Record<'a> {
    /// Decode one record, borrowing every field from `bytes`.
    ///
    /// # Errors
    /// Returns [`StorageError::Corrupt`] for a short, wrongly versioned, or
    /// internally inconsistent record. A record that fails here is real
    /// corruption rather than a torn tail (which the scan never yields), so it
    /// is reported instead of skipped.
    fn decode(bytes: &'a [u8]) -> Result<Self, StorageError> {
        let corrupt = |reason: &str| {
            StorageError::Corrupt(format!("search record: {reason} ({} bytes)", bytes.len()))
        };

        let header = bytes
            .get(..HEADER_LEN)
            .ok_or_else(|| corrupt("truncated header"))?;
        if header[0] != RECORD_VERSION {
            return Err(corrupt(&format!("unsupported version {}", header[0])));
        }
        let flags = header[1];
        // Fail closed on a flag this build does not implement, so a record
        // written by a future version is reported rather than silently read
        // without the guarantee its writer intended.
        if flags & !KNOWN_FLAGS != 0 {
            return Err(corrupt(&format!("unknown flag bits {flags:#04x}")));
        }
        let timestamp = u64::from_le_bytes(
            header[2..10]
                .try_into()
                .map_err(|_| corrupt("truncated timestamp"))?,
        );
        let room_len = u16::from_le_bytes(header[10..12].try_into().expect("2 bytes"));
        let sender_len = u16::from_le_bytes(header[12..14].try_into().expect("2 bytes"));
        let type_len = u16::from_le_bytes(header[14..16].try_into().expect("2 bytes"));
        let event_id_len = u16::from_le_bytes(header[16..18].try_into().expect("2 bytes"));

        let lengths = [room_len, sender_len, type_len, event_id_len];
        let declared: usize = lengths
            .iter()
            .try_fold(HEADER_LEN, |total, len| {
                total.checked_add(usize::from(*len))
            })
            .ok_or_else(|| corrupt("field lengths overflow"))?;
        if declared > bytes.len() {
            return Err(corrupt("declared fields overrun the record"));
        }

        // The body is whatever remains after the four declared fields, so it
        // needs no length of its own.
        let mut cursor = HEADER_LEN;
        let mut take = |len: usize| -> &'a [u8] {
            let end = cursor
                .checked_add(len)
                .expect("declared total fits the record");
            let field = &bytes[cursor..end];
            cursor = end;
            field
        };
        let room_id = take(usize::from(room_len));
        let sender = take(usize::from(sender_len));
        let event_type = take(usize::from(type_len));
        let event_id = std::str::from_utf8(take(usize::from(event_id_len)))
            .map_err(|_| corrupt("event id is not UTF-8"))?;
        let body = bytes
            .get(cursor..)
            .ok_or_else(|| corrupt("truncated body"))?;

        Ok(Self {
            flags,
            timestamp,
            room_id,
            sender,
            event_type,
            event_id,
            body,
        })
    }
}

/// A query's compiled predicates, built once and applied to every record.
struct Matcher<'a> {
    finders: Vec<memmem::Finder<'a>>,
    room_id: Option<&'a [u8]>,
    sender: Option<&'a [u8]>,
    event_type: Option<&'a [u8]>,
    time_range: Option<(u64, u64)>,
}

impl Matcher<'_> {
    fn matches(&self, record: &Record<'_>) -> bool {
        // Removal outranks every predicate, including the absence of all of
        // them: the flag is the authority, so a removed record is skipped even
        // though its cleared fields would otherwise satisfy an empty query.
        if record.flags & FLAG_REMOVED != 0 {
            return false;
        }
        if let Some((start, end)) = self.time_range {
            if record.timestamp < start || record.timestamp > end {
                return false;
            }
        }
        if let Some(room_id) = self.room_id {
            if record.room_id != room_id {
                return false;
            }
        }
        if let Some(sender) = self.sender {
            if record.sender != sender {
                return false;
            }
        }
        if let Some(event_type) = self.event_type {
            if record.event_type != event_type {
                return false;
            }
        }
        self.finders
            .iter()
            .all(|finder| finder.find(record.body).is_some())
    }
}

/// Derive a record's id from its event id.
///
/// Domain-separated and spread, never a constant prefix: `LossyIndex` buckets
/// on the id, so an id that varied only in its low bits would put an entire
/// pack into one bucket.
pub(crate) fn record_id(event_id: &str) -> NodeId {
    let mut hasher = DigestAlgorithm::Blake3.hasher();
    hasher.update(ID_DOMAIN);
    hasher.update(event_id.as_bytes());
    record_logical_id(&hasher.finalize())
}

/// Encode one document into its stored record payload.
///
/// Body is ASCII-lowercased at write time so queries can byte-search without
/// transforming text per record. ASCII-only folding keeps the stored byte
/// length identical to the source, which matters because the body is the last
/// field and its extent is implied by the ones before it. Case variants
/// outside ASCII (Cyrillic, Greek) are therefore not folded.
fn encode(document: &SearchDocument) -> Result<Vec<u8>, StorageError> {
    let body = document
        .body
        .as_deref()
        .unwrap_or_default()
        .to_ascii_lowercase();
    encode_fields(
        0,
        document.timestamp,
        document.room_id.as_bytes(),
        document.sender.as_bytes(),
        document.event_type.as_bytes(),
        document.event_id.as_bytes(),
        body.as_bytes(),
    )
}

/// Encode one record from its already-separated fields.
///
/// Shared by the write path and by [`SearchIndexes::redact`] and
/// [`SearchIndexes::remove`], so a redaction or a removal produces a record in
/// exactly the shape an ordinary write would.
fn encode_fields(
    flags: u8,
    timestamp: u64,
    room_id: &[u8],
    sender: &[u8],
    event_type: &[u8],
    event_id: &[u8],
    body: &[u8],
) -> Result<Vec<u8>, StorageError> {
    let fields: [&[u8]; 5] = [room_id, sender, event_type, event_id, body];

    let mut lengths = [0_u16; 4];
    for (slot, field) in lengths.iter_mut().zip(&fields) {
        *slot = u16::try_from(field.len()).map_err(|_| {
            StorageError::Corrupt(format!(
                "search field too long to encode: {} bytes",
                field.len()
            ))
        })?;
    }

    let capacity = fields
        .iter()
        .try_fold(HEADER_LEN, |total, field| total.checked_add(field.len()))
        .ok_or_else(|| StorageError::Corrupt("search record length overflow".to_owned()))?;

    let mut out = Vec::with_capacity(capacity);
    out.push(RECORD_VERSION);
    debug_assert!(
        flags & !KNOWN_FLAGS == 0,
        "only flags the decoder accepts may be written"
    );
    out.push(flags);
    out.extend_from_slice(&timestamp.to_le_bytes());
    for length in lengths {
        out.extend_from_slice(&length.to_le_bytes());
    }
    for field in fields {
        out.extend_from_slice(field);
    }
    debug_assert_eq!(
        out.len(),
        capacity,
        "encoded record must match its computed length"
    );
    Ok(out)
}
