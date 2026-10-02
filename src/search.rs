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

use memchr::memmem;

use crate::storage::{DigestAlgorithm, NodeData, NodeId, StorageEngine, StorageError};
use crate::template::{
    derive_collection_id, record_logical_id, CollectionMetadata, FrameIdPolicy, PayloadPolicy,
    RecordIdentityRule, COLLECTION_METADATA_RECORD_ID, MEMBER_NAMESPACE_INTL,
};
use crate::PackfileStorage;

/// Version byte on every search record, so a pack written by a different
/// encoding is reported rather than silently mis-decoded.
const RECORD_VERSION: u8 = 1;

/// Length of a record's fixed header: version, timestamp, and four field
/// lengths. The timestamp sits early so a time-bounded query can reject a
/// record without touching any variable-length field.
const HEADER_LEN: usize = 1 + 8 + 4 * 2;

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
    /// Maximum results. `0` means no limit.
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
    /// # Errors
    /// Returns [`StorageError::Io`] on write failure, or [`StorageError::Corrupt`]
    /// if a field is too long to encode.
    pub fn index(&self, document: &SearchDocument) -> Result<(), StorageError> {
        let node_id = record_id(&document.event_id);
        let data = NodeData::new(bytes::Bytes::from(encode(document)?));
        self.engine.create_or_upsert_established_validated(
            &self.collection_id,
            &self.metadata(),
            &node_id,
            &data,
            &mut |_existing| Ok(()),
        )
    }

    /// Add several documents.
    ///
    /// Each document goes through the same upsert as [`Self::index`], so this
    /// is correct for re-ingest but does not coalesce writes into one batch.
    /// Returns the number of documents accepted.
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

    /// Drop the whole search pack. The primary event store is untouched, so
    /// this only discards derived data.
    ///
    /// # Errors
    /// Returns [`StorageError::Io`] on I/O failure.
    pub fn clear(&self) -> Result<(), StorageError> {
        self.engine.delete_collection(&self.collection_id)
    }

    /// Return the event ids of every indexed document satisfying `query`.
    ///
    /// Results are sorted, so output is deterministic despite the scan yielding
    /// records in unspecified order. `limit` is applied after sorting.
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

        let scan = self.engine.scan_collection(&self.collection_id)?;
        let mut hits: Vec<String> = Vec::new();
        for entry in scan {
            let (node_id, data) = entry?;
            // The collection's genesis metadata record lives in the same
            // collection but is not a search record.
            if node_id == COLLECTION_METADATA_RECORD_ID {
                continue;
            }
            let record = Record::decode(&data.bytes)?;
            if matcher.matches(&record) {
                hits.push(record.event_id.to_owned());
            }
        }

        hits.sort_unstable();
        if query.limit != 0 && hits.len() > query.limit {
            hits.truncate(query.limit);
        }
        Ok(hits)
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

/// A decoded search record, borrowing its fields out of the stored payload.
struct Record<'a> {
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
        let timestamp = u64::from_le_bytes(
            header[1..9]
                .try_into()
                .map_err(|_| corrupt("truncated timestamp"))?,
        );
        let room_len = u16::from_le_bytes(header[9..11].try_into().expect("2 bytes"));
        let sender_len = u16::from_le_bytes(header[11..13].try_into().expect("2 bytes"));
        let type_len = u16::from_le_bytes(header[13..15].try_into().expect("2 bytes"));
        let event_id_len = u16::from_le_bytes(header[15..17].try_into().expect("2 bytes"));

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
fn record_id(event_id: &str) -> NodeId {
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

    let fields: [&[u8]; 5] = [
        document.room_id.as_bytes(),
        document.sender.as_bytes(),
        document.event_type.as_bytes(),
        document.event_id.as_bytes(),
        body.as_bytes(),
    ];

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
    out.extend_from_slice(&document.timestamp.to_le_bytes());
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
