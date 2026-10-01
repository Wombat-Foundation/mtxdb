//! Rebuildable secondary indexes for event search.
//!
//! The primary event store remains authoritative.  This module stores only
//! compact postings in named [`AuxiliaryIndex`] collections, so indexes can be
//! dropped and rebuilt without changing event identity or retention policy.

use std::collections::BTreeSet;

use crate::storage::StorageError;
use crate::{AuxiliaryIndex, StorageEngine};

/// A searchable event supplied by the ingestion layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchDocument {
    pub event_id: String,
    pub room_id: String,
    pub sender: String,
    pub event_type: String,
    pub timestamp: u64,
    pub body: Option<String>,
}

/// Search predicates. Empty terms match all indexed events.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SearchQuery {
    pub terms: Vec<String>,
    pub room_id: Option<String>,
    pub sender: Option<String>,
    pub event_type: Option<String>,
    pub time_range: Option<(u64, u64)>,
    pub limit: usize,
}

/// The three independent postings families used by event search.
pub struct SearchIndexes<'a, S: StorageEngine + ?Sized> {
    inverted: AuxiliaryIndex<'a, S>,
    fields: AuxiliaryIndex<'a, S>,
    timestamps: AuxiliaryIndex<'a, S>,
}

impl<'a, S: StorageEngine + ?Sized> SearchIndexes<'a, S> {
    #[must_use]
    pub fn open(engine: &'a S, namespace: &str) -> Self {
        Self {
            inverted: AuxiliaryIndex::open(engine, &format!("{namespace}:inverted")),
            fields: AuxiliaryIndex::open(engine, &format!("{namespace}:fields")),
            timestamps: AuxiliaryIndex::open(engine, &format!("{namespace}:timestamps")),
        }
    }

    /// Add one document. Re-indexing the same event is idempotent.
    pub fn index(&self, document: &SearchDocument) -> Result<(), StorageError> {
        self.add_posting(&self.inverted, "__all__", &document.event_id)?;
        for term in document.body.as_deref().map(tokenize).unwrap_or_default() {
            self.add_posting(&self.inverted, &term, &document.event_id)?;
        }
        for (field, value) in [
            ("room", &document.room_id),
            ("sender", &document.sender),
            ("type", &document.event_type),
        ] {
            self.add_posting(
                &self.fields,
                &format!("{field}:{value}"),
                &document.event_id,
            )?;
        }
        self.add_posting(
            &self.timestamps,
            &format!("t:{}", document.timestamp),
            &document.event_id,
        )
    }

    /// Return event IDs satisfying all predicates. Results are deterministic.
    pub fn search(&self, query: &SearchQuery) -> Result<Vec<String>, StorageError> {
        let mut result: Option<BTreeSet<String>> = None;
        let mut keys = query
            .terms
            .iter()
            .map(|t| t.to_ascii_lowercase())
            .collect::<Vec<_>>();
        if let Some(value) = &query.room_id {
            keys.push(format!("room:{value}"));
        }
        if let Some(value) = &query.sender {
            keys.push(format!("sender:{value}"));
        }
        if let Some(value) = &query.event_type {
            keys.push(format!("type:{value}"));
        }
        if let Some((start, end)) = query.time_range {
            let mut times = BTreeSet::new();
            for timestamp in start..=end {
                times.extend(decode(
                    self.timestamps.get(format!("t:{timestamp}").as_bytes())?,
                ));
            }
            result = Some(times);
        }
        if keys.is_empty() && result.is_none() {
            result = Some(decode(self.inverted.get(b"__all__")?).into_iter().collect());
        }
        for key in keys {
            let postings = decode(
                self.inverted
                    .get(key.as_bytes())?
                    .or(self.fields.get(key.as_bytes())?),
            );
            let set = postings.into_iter().collect::<BTreeSet<_>>();
            result =
                Some(result.map_or(set.clone(), |old| old.intersection(&set).cloned().collect()));
        }
        let mut out: Vec<String> = result.unwrap_or_default().into_iter().collect();
        if query.limit != 0 {
            out.truncate(query.limit);
        }
        Ok(out)
    }

    fn add_posting(
        &self,
        index: &AuxiliaryIndex<'a, S>,
        key: &str,
        event_id: &str,
    ) -> Result<(), StorageError> {
        let mut values = decode(index.get(key.as_bytes())?);
        if !values.iter().any(|v| v == event_id) {
            values.push(event_id.to_owned());
            values.sort();
        }
        index.put(key.as_bytes(), &encode(&values))
    }
}

fn tokenize(body: &str) -> Vec<String> {
    body.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_ascii_lowercase)
        .collect()
}
fn encode(values: &[String]) -> Vec<u8> {
    values.join("\0").into_bytes()
}
fn decode(value: Option<Vec<u8>>) -> Vec<String> {
    value
        .unwrap_or_default()
        .split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .filter_map(|s| String::from_utf8(s.to_vec()).ok())
        .collect()
}
