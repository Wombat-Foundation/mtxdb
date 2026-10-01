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
        (vec!["$p"], vec!["$a"], None),
    ] {
        let error = adj
            .record_event(&db, "$e", &prev, &auth, relation)
            .unwrap_err();
        assert!(matches!(error, StorageError::Collision(_)), "{error}");
    }
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
    assert!(matches!(error, StorageError::Internal(_)), "{error}");
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
