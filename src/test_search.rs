//! Tests for the search pack: dense pre-extracted records scanned per query.
//!
//! These run against a real on-disk [`PackfileStorage`] rather than an
//! in-memory engine, because the query path is a pack scan — the thing worth
//! testing is the behavior a caller actually gets.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::storage::{NodeData, StorageEngine, StorageError};
use crate::{PackfileStorage, SearchDocument, SearchIndexes, SearchQuery};

fn test_dir(name: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "mdb_test_search_{name}_{}_{id}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create test dir");
    dir
}

fn store(name: &str) -> (PathBuf, PackfileStorage) {
    let dir = test_dir(name);
    let store = PackfileStorage::open(dir.clone()).expect("open store");
    (dir, store)
}

fn doc(
    event_id: &str,
    room: &str,
    sender: &str,
    kind: &str,
    ts: u64,
    body: Option<&str>,
) -> SearchDocument {
    SearchDocument {
        event_id: event_id.to_owned(),
        room_id: room.to_owned(),
        sender: sender.to_owned(),
        event_type: kind.to_owned(),
        timestamp: ts,
        body: body.map(str::to_owned),
    }
}

fn terms<'q>(query: &'q mut SearchQuery, values: &[&str]) -> &'q mut SearchQuery {
    query.terms = values.iter().map(|v| (*v).to_owned()).collect();
    query
}

#[test]
fn empty_pack_searches_empty_without_error() {
    let (_dir, store) = store("empty_pack");
    let index = SearchIndexes::open(&store, "ns");
    assert_eq!(index.len().expect("len"), None);
    assert!(index
        .search(&SearchQuery::default())
        .expect("search")
        .is_empty());
}

#[test]
fn finds_substring_and_indexes_by_event_id() {
    let (_dir, store) = store("basic");
    let index = SearchIndexes::open(&store, "ns");
    index
        .index(&doc(
            "$a",
            "!r:example.org",
            "@u:example.org",
            "m.room.message",
            1_000,
            Some("the quick brown fox"),
        ))
        .expect("index");
    index
        .index(&doc(
            "$b",
            "!r:example.org",
            "@u:example.org",
            "m.room.message",
            2_000,
            Some("jumps over the lazy dog"),
        ))
        .expect("index");

    let mut query = SearchQuery::default();
    assert_eq!(
        index.search(terms(&mut query, &["brown"])).expect("search"),
        vec!["$a"]
    );
    // Mid-word and multi-term AND both work.
    assert_eq!(
        index.search(terms(&mut query, &["ump"])).expect("search"),
        vec!["$b"]
    );
    assert_eq!(
        index
            .search(terms(&mut query, &["the", "dog"]))
            .expect("search"),
        vec!["$b"]
    );
    // "the" is in both bodies.
    assert_eq!(
        index.search(terms(&mut query, &["the"])).expect("search"),
        vec!["$a", "$b"]
    );
}

#[test]
fn matching_is_ascii_case_insensitive() {
    let (_dir, store) = store("case");
    let index = SearchIndexes::open(&store, "ns");
    index
        .index(&doc(
            "$a",
            "!r:example.org",
            "@u:example.org",
            "m.room.message",
            1_000,
            Some("Hello World"),
        ))
        .expect("index");

    let mut query = SearchQuery::default();
    for needle in ["hello", "HELLO", "HeLlO", "hello world"] {
        assert_eq!(
            index.search(terms(&mut query, &[needle])).expect("search"),
            vec!["$a"],
            "needle {needle}"
        );
    }
}

#[test]
fn non_ascii_body_is_searchable_verbatim() {
    let (_dir, store) = store("unicode");
    let index = SearchIndexes::open(&store, "ns");
    index
        .index(&doc(
            "$a",
            "!r:example.org",
            "@u:example.org",
            "m.room.message",
            1_000,
            Some("héllo wörld"),
        ))
        .expect("index");
    index
        .index(&doc(
            "$b",
            "!r:example.org",
            "@u:example.org",
            "m.room.message",
            2_000,
            Some("日本語のテキスト"),
        ))
        .expect("index");

    let mut query = SearchQuery::default();
    assert_eq!(
        index.search(terms(&mut query, &["wörld"])).expect("search"),
        vec!["$a"]
    );
    assert_eq!(
        index
            .search(terms(&mut query, &["テキスト"]))
            .expect("search"),
        vec!["$b"]
    );
    // ASCII-only folding: non-ASCII case variants are deliberately not folded.
    assert!(index
        .search(terms(&mut query, &["WÖRLD"]))
        .expect("search")
        .is_empty());
}

#[test]
fn body_is_the_implied_trailing_field() {
    let (_dir, store) = store("implied_body");
    let index = SearchIndexes::open(&store, "ns");
    // A body containing bytes that look like delimiters must not shift the
    // field boundaries: the body is whatever follows the four declared fields.
    index
        .index(&doc(
            "$a",
            "!r:example.org",
            "@u:example.org",
            "m.room.message",
            1_000,
            Some("nul\u{0}ish \\u{1} padding tail"),
        ))
        .expect("index");

    let mut query = SearchQuery::default();
    assert_eq!(
        index
            .search(terms(&mut query, &["padding tail"]))
            .expect("search"),
        vec!["$a"]
    );
    // The filter fields still resolve correctly alongside the odd body.
    query.terms.clear();
    query.room_id = Some("!r:example.org".to_owned());
    assert_eq!(index.search(&query).expect("search"), vec!["$a"]);
}

#[test]
fn header_filters_narrow_results() {
    let (_dir, store) = store("filters");
    let index = SearchIndexes::open(&store, "ns");
    index
        .index(&doc(
            "$a",
            "!r:one",
            "@u:alice",
            "m.room.message",
            1_000,
            Some("shared token"),
        ))
        .expect("index");
    index
        .index(&doc(
            "$b",
            "!r:two",
            "@u:bob",
            "m.room.member",
            2_000,
            Some("shared token"),
        ))
        .expect("index");
    index
        .index(&doc(
            "$c",
            "!r:one",
            "@u:bob",
            "m.room.message",
            3_000,
            Some("shared token"),
        ))
        .expect("index");

    let mut query = SearchQuery::default();
    terms(&mut query, &["shared"]);
    assert_eq!(
        index.search(&query).expect("search"),
        vec!["$a", "$b", "$c"]
    );

    query.room_id = Some("!r:one".to_owned());
    assert_eq!(index.search(&query).expect("search"), vec!["$a", "$c"]);

    query.room_id = None;
    query.sender = Some("@u:bob".to_owned());
    assert_eq!(index.search(&query).expect("search"), vec!["$b", "$c"]);

    query.sender = None;
    query.event_type = Some("m.room.message".to_owned());
    assert_eq!(index.search(&query).expect("search"), vec!["$a", "$c"]);
}

/// Every filter narrows independently *and* in combination: a record must
/// satisfy all of them at once.
#[test]
fn filters_compose_as_a_conjunction() {
    let (_dir, store) = store("filter_conjunction");
    let index = SearchIndexes::open(&store, "ns");
    index
        .index(&doc(
            "$a",
            "!r:one",
            "@u:alice",
            "m.room.message",
            1_000,
            Some("shared token"),
        ))
        .expect("index");
    index
        .index(&doc(
            "$b",
            "!r:two",
            "@u:bob",
            "m.room.member",
            2_000,
            Some("shared token"),
        ))
        .expect("index");
    index
        .index(&doc(
            "$c",
            "!r:one",
            "@u:bob",
            "m.room.message",
            3_000,
            Some("shared token"),
        ))
        .expect("index");

    let mut query = SearchQuery::default();
    terms(&mut query, &["shared"]);
    query.sender = Some("@u:bob".to_owned());
    query.event_type = Some("m.room.member".to_owned());
    assert_eq!(index.search(&query).expect("search"), vec!["$b"]);

    query.event_type = None;
    query.room_id = Some("!r:one".to_owned());
    assert_eq!(index.search(&query).expect("search"), vec!["$c"]);

    // alice only ever sent a message, so this conjunction is empty.
    query.sender = Some("@u:alice".to_owned());
    query.event_type = Some("m.room.member".to_owned());
    assert!(index.search(&query).expect("search").is_empty());
}

#[test]
fn time_range_bounds_are_inclusive() {
    let (_dir, store) = store("time_range");
    let index = SearchIndexes::open(&store, "ns");
    for (n, ts) in [1_000_u64, 2_000, 3_000].into_iter().enumerate() {
        index
            .index(&doc(
                &format!("${n}"),
                "!r:one",
                "@u:a",
                "m.room.message",
                ts,
                Some("tick"),
            ))
            .expect("index");
    }

    let mut query = SearchQuery::default();
    terms(&mut query, &["tick"]);
    query.time_range = Some((2_000, 3_000));
    assert_eq!(index.search(&query).expect("search"), vec!["$1", "$2"]);

    // A single-instant range is a valid inclusive range.
    query.time_range = Some((2_000, 2_000));
    assert_eq!(index.search(&query).expect("search"), vec!["$1"]);

    // Inverted and disjoint ranges simply match nothing.
    query.time_range = Some((3_000, 1_000));
    assert!(index.search(&query).expect("search").is_empty());
    query.time_range = Some((9_000, 9_500));
    assert!(index.search(&query).expect("search").is_empty());
}

#[test]
fn empty_query_returns_everything_and_limit_truncates() {
    let (_dir, store) = store("limit");
    let index = SearchIndexes::open(&store, "ns");
    for n in 0_u64..5 {
        index
            .index(&doc(
                &format!("${n}"),
                "!r:one",
                "@u:a",
                "m.room.message",
                n,
                Some("any"),
            ))
            .expect("index");
    }

    let mut query = SearchQuery::default();
    assert_eq!(index.search(&query).expect("search").len(), 5);
    query.limit = 2;
    // Sorted, so the truncation is stable rather than scan-order dependent.
    assert_eq!(index.search(&query).expect("search"), vec!["$0", "$1"]);
}

#[test]
fn reindexing_an_unchanged_document_is_idempotent() {
    let (_dir, store) = store("idempotent");
    let index = SearchIndexes::open(&store, "ns");
    let document = doc(
        "$a",
        "!r:one",
        "@u:a",
        "m.room.message",
        1_000,
        Some("stable"),
    );

    index.index(&document).expect("index");
    let after_first = index.len().expect("len").expect("established");
    index.index(&document).expect("reindex");
    let after_second = index.len().expect("len").expect("established");

    assert_eq!(
        after_first, after_second,
        "an unchanged re-index must not add a record"
    );
    let mut query = SearchQuery::default();
    assert_eq!(
        index
            .search(terms(&mut query, &["stable"]))
            .expect("search"),
        vec!["$a"]
    );
}

#[test]
fn reindexing_an_edited_document_replaces_rather_than_duplicates() {
    let (_dir, store) = store("replace");
    let index = SearchIndexes::open(&store, "ns");
    index
        .index(&doc(
            "$a",
            "!r:one",
            "@u:a",
            "m.room.message",
            1_000,
            Some("original text"),
        ))
        .expect("index");
    index
        .index(&doc(
            "$a",
            "!r:one",
            "@u:a",
            "m.room.message",
            1_000,
            Some("edited text"),
        ))
        .expect("reindex");

    let mut query = SearchQuery::default();
    assert_eq!(
        index
            .search(terms(&mut query, &["original"]))
            .expect("search"),
        Vec::<String>::new()
    );
    assert_eq!(
        index
            .search(terms(&mut query, &["edited"]))
            .expect("search"),
        vec!["$a"]
    );
    // Still exactly one indexed application record.
    assert_eq!(
        index.len().expect("len").expect("established"),
        2,
        "one application record plus genesis metadata"
    );
}

#[test]
fn documents_without_a_body_are_still_searchable_by_header() {
    let (_dir, store) = store("no_body");
    let index = SearchIndexes::open(&store, "ns");
    index
        .index(&doc(
            "$a",
            "!r:one",
            "@u:a",
            "m.room.encrypted",
            1_000,
            None,
        ))
        .expect("index");

    let mut query = SearchQuery {
        room_id: Some("!r:one".to_owned()),
        ..SearchQuery::default()
    };
    assert_eq!(index.search(&query).expect("search"), vec!["$a"]);

    // No body means no text to match.
    assert!(index
        .search(terms(&mut query, &["anything"]))
        .expect("search")
        .is_empty());
}

#[test]
fn index_many_accepts_a_batch() {
    let (_dir, store) = store("many");
    let index = SearchIndexes::open(&store, "ns");
    let documents: Vec<SearchDocument> = (0_u64..4)
        .map(|n| {
            doc(
                &format!("${n}"),
                "!r:one",
                "@u:a",
                "m.room.message",
                n,
                Some("batchy"),
            )
        })
        .collect();
    assert_eq!(index.index_many(&documents).expect("index_many"), 4);

    let mut query = SearchQuery::default();
    assert_eq!(
        index
            .search(terms(&mut query, &["batchy"]))
            .expect("search")
            .len(),
        4
    );
}

#[test]
fn clear_drops_the_pack_without_touching_other_namespaces() {
    let (_dir, store) = store("clear");
    let index = SearchIndexes::open(&store, "ns");
    let other = SearchIndexes::open(&store, "other");
    index
        .index(&doc(
            "$a",
            "!r:one",
            "@u:a",
            "m.room.message",
            1_000,
            Some("gone soon"),
        ))
        .expect("index");
    other
        .index(&doc(
            "$b",
            "!r:one",
            "@u:a",
            "m.room.message",
            1_000,
            Some("still here"),
        ))
        .expect("index");

    index.clear().expect("clear");

    let mut query = SearchQuery::default();
    assert!(index
        .search(terms(&mut query, &["gone"]))
        .expect("search")
        .is_empty());
    assert_eq!(
        other.search(terms(&mut query, &["still"])).expect("search"),
        vec!["$b"]
    );
}

#[test]
fn namespaces_are_isolated() {
    let (_dir, store) = store("namespaces");
    let a = SearchIndexes::open(&store, "alpha");
    let b = SearchIndexes::open(&store, "beta");
    assert_ne!(a.collection_id(), b.collection_id());
    a.index(&doc(
        "$a",
        "!r:one",
        "@u:a",
        "m.room.message",
        1_000,
        Some("alpha only"),
    ))
    .expect("index");

    let mut query = SearchQuery::default();
    assert_eq!(
        a.search(terms(&mut query, &["alpha"])).expect("search"),
        vec!["$a"]
    );
    assert!(b
        .search(terms(&mut query, &["alpha"]))
        .expect("search")
        .is_empty());
}

#[test]
fn a_corrupt_record_is_reported_rather_than_skipped() {
    let (_dir, store) = store("corrupt");
    let index = SearchIndexes::open(&store, "ns");
    index
        .index(&doc(
            "$a",
            "!r:one",
            "@u:a",
            "m.room.message",
            1_000,
            Some("fine"),
        ))
        .expect("index");

    // Write a record with a bogus version byte straight past the index API.
    let corrupt_id = [0xAB_u8; 16];
    store
        .put(
            &index.collection_id(),
            &corrupt_id,
            &NodeData::new(bytes::Bytes::from_static(b"not a search record")),
        )
        .expect("put corrupt");

    let error = index
        .search(&SearchQuery::default())
        .expect_err("must reject a corrupt record");
    assert!(
        matches!(&error, StorageError::Corrupt(message) if message.contains("search record")),
        "expected a search-record corruption error, got {error:?}"
    );
}

#[test]
fn a_record_whose_lengths_overrun_its_payload_is_rejected() {
    let (_dir, store) = store("overrun");
    let index = SearchIndexes::open(&store, "ns");

    // Valid header shape, but the declared room length runs past the payload.
    let mut record = vec![1_u8];
    record.extend_from_slice(&7_u64.to_le_bytes());
    record.extend_from_slice(&u16::MAX.to_le_bytes());
    record.extend_from_slice(&0_u16.to_le_bytes());
    record.extend_from_slice(&0_u16.to_le_bytes());
    record.extend_from_slice(&0_u16.to_le_bytes());
    record.extend_from_slice(b"short");

    store
        .put(
            &index.collection_id(),
            &[0xCD_u8; 16],
            &NodeData::new(bytes::Bytes::from(record)),
        )
        .expect("put");

    let error = index
        .search(&SearchQuery::default())
        .expect_err("must reject an overrunning record");
    assert!(
        matches!(&error, StorageError::Corrupt(message) if message.contains("overrun")),
        "expected an overrun corruption error, got {error:?}"
    );
}
