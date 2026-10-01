#![cfg(test)]

use super::matrix_adjacency::*;
use crate::database::Database;
use crate::layout::ShardType;
use crate::storage::StorageError;
use std::path::PathBuf;

const ROOM: &str = "!room:example.org";

fn test_root(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mtxdb-mtxadj-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn adjacency() -> MatrixAdjacency {
    MatrixAdjacency::new(ShardType::Edges, ROOM)
}

#[test]
fn records_and_reads_prev_auth_and_relation() {
    let root = test_root("roundtrip");
    let db = Database::open(root.clone()).unwrap();
    let adj = adjacency();
    adj.record_event(&db, "$create", &[], &[], None).unwrap();
    adj.record_event(&db, "$msg", &["$create"], &["$create"], None)
        .unwrap();
    adj.record_event(
        &db,
        "$react",
        &["$msg"],
        &["$create"],
        Some(RelationRef {
            target: "$msg",
            rel_type: "m.annotation",
        }),
    )
    .unwrap();

    assert_eq!(
        adj.prev_of(&db, "$react").unwrap(),
        Some(vec!["$msg".to_owned()])
    );
    assert_eq!(
        adj.auth_of(&db, "$react").unwrap(),
        Some(vec!["$create".to_owned()])
    );
    assert_eq!(
        adj.prev_of(&db, "$create").unwrap(),
        Some(vec![]),
        "known leaf"
    );
    assert_eq!(adj.prev_of(&db, "$never-seen").unwrap(), None);
    assert_eq!(
        adj.relation_of(&db, "$react", &AlwaysVisible).unwrap(),
        Some(Relation {
            target: "$msg".to_owned(),
            rel_type: "m.annotation".to_owned()
        })
    );
    assert_eq!(adj.relation_of(&db, "$msg", &AlwaysVisible).unwrap(), None);
    assert!(adj.verify(&db).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn recording_is_idempotent_but_an_event_never_changes() {
    let root = test_root("idempotent");
    let db = Database::open(root.clone()).unwrap();
    let adj = adjacency();
    let relation = Some(RelationRef {
        target: "$t",
        rel_type: "m.thread",
    });
    let id = adj
        .record_event(&db, "$e", &["$p"], &["$a"], relation)
        .unwrap();
    assert_eq!(
        adj.record_event(&db, "$e", &["$p"], &["$a"], relation)
            .unwrap(),
        id
    );
    for (prev, auth, relation) in [
        (vec!["$other"], vec!["$a"], relation),
        (vec!["$p"], vec!["$other"], relation),
        (
            vec!["$p"],
            vec!["$a"],
            Some(RelationRef {
                target: "$t",
                rel_type: "m.replace",
            }),
        ),
    ] {
        let error = adj
            .record_event(&db, "$e", &prev, &auth, relation)
            .unwrap_err();
        assert!(matches!(error, StorageError::Collision(_)), "{error}");
    }
    // Recording again with no relation is not a change: no relation known leaves
    // the stored one alone (see `a_relation_can_be_added_later_but_never_changed_or_removed`).
    assert_eq!(
        adj.record_event(&db, "$e", &["$p"], &["$a"], None).unwrap(),
        id
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn spec_relation_types_have_fixed_ids_regardless_of_import_order() {
    let ids = |name: &str, first: &[&str]| {
        let root = test_root(name);
        let db = Database::open(root.clone()).unwrap();
        let adj = adjacency();
        for rel_type in first {
            adj.kind_id(&db, rel_type).unwrap();
        }
        let known: Vec<u16> = KNOWN_RELATION_TYPES
            .iter()
            .map(|t| adj.kind_id(&db, t).unwrap())
            .collect();
        drop(db);
        let _ = std::fs::remove_dir_all(&root);
        known
    };
    // A custom type seen first must not displace the spec types.
    let plain = ids("kinds-plain", &[]);
    let custom_first = ids("kinds-custom-first", &["org.example.custom", "m.thread"]);
    let reversed = ids("kinds-reversed", &["m.thread", "m.annotation"]);
    assert_eq!(plain, vec![1, 2, 3, 4]);
    assert_eq!(custom_first, plain);
    assert_eq!(reversed, plain);
}

#[test]
fn unknown_relation_types_are_preserved_and_ids_are_stable_across_reopen() {
    let root = test_root("unknown");
    let (custom, other) = {
        let db = Database::open(root.clone()).unwrap();
        let adj = adjacency();
        let custom = adj.kind_id(&db, "org.example.custom").unwrap();
        let other = adj.kind_id(&db, "org.example.other").unwrap();
        assert!(
            custom > 4 && other > custom,
            "allocated after the spec types"
        );
        adj.record_event(
            &db,
            "$e",
            &[],
            &[],
            Some(RelationRef {
                target: "$t",
                rel_type: "org.example.custom",
            }),
        )
        .unwrap();
        (custom, other)
    };
    let db = Database::open(root.clone()).unwrap();
    let adj = adjacency();
    // Ids never change once assigned, however the types are asked for later.
    assert_eq!(adj.kind_id(&db, "org.example.other").unwrap(), other);
    assert_eq!(adj.kind_id(&db, "org.example.custom").unwrap(), custom);
    assert_eq!(
        adj.relation_type(&db, custom).unwrap().as_deref(),
        Some("org.example.custom")
    );
    // The unknown type survives in the stored relation, not dropped or renamed.
    assert_eq!(
        adj.relation_of(&db, "$e", &AlwaysVisible)
            .unwrap()
            .unwrap()
            .rel_type,
        "org.example.custom"
    );
    assert!(adj.verify(&db).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn visibility_filter_hides_relations_only() {
    let root = test_root("visibility");
    let db = Database::open(root.clone()).unwrap();
    let adj = adjacency();
    adj.record_event(
        &db,
        "$react",
        &["$msg"],
        &["$create"],
        Some(RelationRef {
            target: "$msg",
            rel_type: "m.annotation",
        }),
    )
    .unwrap();
    let hidden = |event_id: &str| event_id != "$react";
    assert_eq!(adj.relation_of(&db, "$react", &hidden).unwrap(), None);
    // prev/auth are signed core fields redaction never changes: not filtered.
    assert_eq!(
        adj.prev_of(&db, "$react").unwrap(),
        Some(vec!["$msg".to_owned()])
    );
    assert_eq!(
        adj.auth_of(&db, "$react").unwrap(),
        Some(vec!["$create".to_owned()])
    );
    // Visibility is external and reversible (e.g. un-reject): nothing stored.
    assert!(adj
        .relation_of(&db, "$react", &AlwaysVisible)
        .unwrap()
        .is_some());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn relation_changes_never_touch_the_auth_adjacency() {
    let root = test_root("auth-isolation");
    let db = Database::open(root.clone()).unwrap();
    let adj = adjacency();
    adj.record_event(&db, "$a", &[], &["$create"], None)
        .unwrap();
    adj.record_event(
        &db,
        "$b",
        &[],
        &["$a"],
        Some(RelationRef {
            target: "$a",
            rel_type: "m.reference",
        }),
    )
    .unwrap();
    // `$b` relates to `$a` but its auth chain is only what it declared.
    assert_eq!(adj.auth_of(&db, "$b").unwrap(), Some(vec!["$a".to_owned()]));
    assert_eq!(
        adj.auth_of(&db, "$a").unwrap(),
        Some(vec!["$create".to_owned()])
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn relation_kind_space_is_a_hard_u16_limit() {
    let root = test_root("kind-limit");
    let db = Database::open(root.clone()).unwrap();
    let adj = adjacency();
    adj.kind_id(&db, "m.thread").unwrap();
    adj.set_kind_counter_for_test(&db, u32::from(u16::MAX))
        .unwrap();
    // The last valid id is allocatable; the next type is refused, not wrapped.
    assert_eq!(adj.kind_id(&db, "org.example.last").unwrap(), u16::MAX);
    let error = adj.kind_id(&db, "org.example.one-too-many").unwrap_err();
    assert!(error.is_exhausted(), "{error}");
    assert_eq!(adj.kind_id(&db, "org.example.last").unwrap(), u16::MAX);
    // Nothing was published past the limit: the refused type has no id at all.
    assert_eq!(
        adj.kinds_lookup_for_test(&db, "org.example.one-too-many"),
        None
    );
    // (No `verify` here: parking the counter leaves an artificial gap of ids.)
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn purge_removes_events_and_the_kind_dictionary_together() {
    let root = test_root("purge");
    let db = Database::open(root.clone()).unwrap();
    let adj = adjacency();
    adj.record_event(
        &db,
        "$e",
        &[],
        &[],
        Some(RelationRef {
            target: "$t",
            rel_type: "org.example.custom",
        }),
    )
    .unwrap();
    let other_room = MatrixAdjacency::new(ShardType::Edges, "!other:example.org");
    other_room.record_event(&db, "$x", &[], &[], None).unwrap();

    adj.purge(&db).unwrap();
    assert_eq!(adj.short_id(&db, "$e").unwrap(), None);
    assert_eq!(
        adj.relation_type(&db, 5).unwrap(),
        None,
        "dictionary purged too"
    );
    assert!(
        other_room.short_id(&db, "$x").unwrap().is_some(),
        "other rooms untouched"
    );
    // The purged room starts over, including the fixed spec ids.
    assert_eq!(adj.kind_id(&db, "m.annotation").unwrap(), 1);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// A batch with relations stores exactly what recording one event at a time
/// would, including the relation-kind dictionary shared across the batch.
#[test]
fn a_batch_matches_recording_events_one_at_a_time() {
    let batched_root = test_root("batch-vs-single-a");
    let single_root = test_root("batch-vs-single-b");
    let batched_db = Database::open(batched_root.clone()).unwrap();
    let single_db = Database::open(single_root.clone()).unwrap();
    let batched = adjacency();
    let single = adjacency();

    let events = [
        EventRecord {
            event_id: "$create",
            prev: &[],
            auth: &[],
            relation: None,
        },
        EventRecord {
            event_id: "$msg",
            prev: &["$create"],
            auth: &["$create"],
            relation: None,
        },
        EventRecord {
            event_id: "$react1",
            prev: &["$msg"],
            auth: &["$create"],
            relation: Some(RelationRef {
                target: "$msg",
                rel_type: "m.annotation",
            }),
        },
        EventRecord {
            event_id: "$react2",
            prev: &["$react1"],
            auth: &["$create"],
            relation: Some(RelationRef {
                target: "$msg",
                rel_type: "m.annotation",
            }),
        },
        EventRecord {
            event_id: "$edit",
            prev: &["$react2"],
            auth: &["$create"],
            relation: Some(RelationRef {
                target: "$msg",
                rel_type: "org.example.custom",
            }),
        },
    ];

    let ids = batched.record_events(&batched_db, &events).unwrap();
    for event in &events {
        single
            .record_event(
                &single_db,
                event.event_id,
                event.prev,
                event.auth,
                event.relation,
            )
            .unwrap();
    }
    assert_eq!(ids.len(), events.len());
    for event in &events {
        assert_eq!(
            batched.prev_of(&batched_db, event.event_id).unwrap(),
            single.prev_of(&single_db, event.event_id).unwrap(),
            "{}",
            event.event_id
        );
        assert_eq!(
            batched.auth_of(&batched_db, event.event_id).unwrap(),
            single.auth_of(&single_db, event.event_id).unwrap(),
            "{}",
            event.event_id
        );
        assert_eq!(
            batched
                .relation_of(&batched_db, event.event_id, &AlwaysVisible)
                .unwrap(),
            single
                .relation_of(&single_db, event.event_id, &AlwaysVisible)
                .unwrap(),
            "{}",
            event.event_id
        );
    }
    assert_eq!(
        batched.kind_id(&batched_db, "m.annotation").unwrap(),
        single.kind_id(&single_db, "m.annotation").unwrap()
    );
    assert!(batched.verify(&batched_db).unwrap().is_consistent());
    drop(batched_db);
    drop(single_db);
    let _ = std::fs::remove_dir_all(&batched_root);
    let _ = std::fs::remove_dir_all(&single_root);
}

/// A relation can be added to an event recorded without one, is never removed by
/// recording the event again without it (a redacted copy has no `m.relates_to`),
/// and is never replaced by a different one.
#[test]
fn a_relation_can_be_added_later_but_never_changed_or_removed() {
    let root = test_root("relation-late");
    let db = Database::open(root.clone()).unwrap();
    let adj = adjacency();
    let reaction = RelationRef {
        target: "$msg",
        rel_type: "m.annotation",
    };
    let relation = |db: &Database| {
        adj.relation_of(db, "$react", &AlwaysVisible)
            .unwrap()
            .map(|r| (r.target, r.rel_type))
    };

    // First seen redacted: no relation known.
    adj.record_event(&db, "$react", &["$msg"], &["$create"], None)
        .unwrap();
    assert_eq!(relation(&db), None);

    // The unredacted copy arrives: the relation is added, nothing else changes.
    adj.record_event(&db, "$react", &["$msg"], &["$create"], Some(reaction))
        .unwrap();
    assert_eq!(
        relation(&db),
        Some(("$msg".to_owned(), "m.annotation".to_owned()))
    );

    // The redacted copy again: the stored relation is not removed.
    adj.record_event(&db, "$react", &["$msg"], &["$create"], None)
        .unwrap();
    assert_eq!(
        relation(&db),
        Some(("$msg".to_owned(), "m.annotation".to_owned()))
    );

    // A different relation is a collision, and leaves the stored one intact.
    let error = adj
        .record_event(
            &db,
            "$react",
            &["$msg"],
            &["$create"],
            Some(RelationRef {
                target: "$other",
                rel_type: "m.annotation",
            }),
        )
        .unwrap_err();
    assert!(matches!(error, StorageError::Collision(_)), "{error}");
    assert_eq!(
        relation(&db),
        Some(("$msg".to_owned(), "m.annotation".to_owned()))
    );

    // The batch path follows the same rule.
    let batch = [EventRecord {
        event_id: "$react",
        prev: &["$msg"],
        auth: &["$create"],
        relation: None,
    }];
    adj.record_events(&db, &batch).unwrap();
    assert_eq!(
        relation(&db),
        Some(("$msg".to_owned(), "m.annotation".to_owned()))
    );
    assert!(adj.verify(&db).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}
