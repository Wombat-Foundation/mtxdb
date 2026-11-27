use super::csr_adjacency::*;
use crate::database::Database;
use crate::storage::StorageError;
use std::path::PathBuf;

const ROOM: &str = "!room:example.org";

fn test_root(name: &str) -> PathBuf {
    let path =
        crate::test_scratch::root().join(format!("mtxdb-mtxadj-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn adjacency() -> CsrAdjacency {
    CsrAdjacency::new(ROOM)
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
fn referenced_event_ids_union_and_split_by_family() {
    let root = test_root("referenced-ids");
    let db = Database::open(root.clone()).unwrap();
    let adj = adjacency();
    adj.record_event(&db, "$create", &[], &[], None).unwrap();
    adj.record_event(&db, "$msg", &["$create"], &["$create"], None)
        .unwrap();
    adj.record_event(
        &db,
        "$react",
        &["$msg"],
        &["$create", "$missing-auth"],
        None,
    )
    .unwrap();

    let set = |ids: &[&str]| -> std::collections::BTreeSet<String> {
        ids.iter().map(|id| (*id).to_owned()).collect()
    };
    assert_eq!(
        adj.referenced_event_ids(&db).unwrap(),
        set(&["$create", "$msg", "$missing-auth"]),
        "the union names every cited id, whether or not its body exists"
    );
    let (prev, auth) = adj.referenced_event_ids_by_family(&db).unwrap();
    assert_eq!(prev, set(&["$create", "$msg"]));
    assert_eq!(auth, set(&["$create", "$missing-auth"]));
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn resolve_event_ids_round_trips_short_ids_and_reports_absent_ones() {
    let root = test_root("resolve_ids");
    let db = Database::open(root.clone()).unwrap();
    let adj = adjacency();
    adj.record_event(&db, "$create", &[], &[], None).unwrap();
    adj.record_event(&db, "$msg", &["$create"], &["$create"], None)
        .unwrap();
    let create = adj.short_id(&db, "$create").unwrap().unwrap();
    let msg = adj.short_id(&db, "$msg").unwrap().unwrap();
    let absent = u32::MAX;

    assert_eq!(
        adj.resolve_event_ids(&db, &[msg, absent, create]).unwrap(),
        vec![Some("$msg".to_owned()), None, Some("$create".to_owned()),],
        "order is preserved and an unassigned id resolves to None"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn reverse_family_transposes_persisted_rows() {
    let root = test_root("reverse-family");
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

    let create = adj.short_id(&db, "$create").unwrap().unwrap();
    let msg = adj.short_id(&db, "$msg").unwrap().unwrap();
    let react = adj.short_id(&db, "$react").unwrap().unwrap();
    let prev = adj.reverse_family(&db, PREV).unwrap();
    let auth = adj.reverse_family(&db, AUTH).unwrap();
    let relations = adj.reverse_family(&db, RELATIONS).unwrap();

    assert_eq!(prev.child_count(create), Some(1));
    assert_eq!(prev.children_of(create).unwrap()[0].target, msg);
    assert_eq!(prev.children_of(msg).unwrap()[0].target, react);
    assert_eq!(auth.child_count(create), Some(2));
    assert_eq!(
        auth.children_of(create)
            .unwrap()
            .iter()
            .map(|edge| edge.target)
            .collect::<Vec<_>>(),
        vec![msg, react]
    );
    assert_eq!(relations.child_count(msg), Some(1));
    assert_eq!(relations.children_of(msg).unwrap()[0].target, react);
    assert_eq!(
        relations.children_of(msg).unwrap()[0].kind,
        adj.kind_id(&db, "m.annotation").unwrap()
    );

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
    let other_room = CsrAdjacency::new("!other:example.org");
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

#[test]
fn concurrent_recordings_of_one_event_cannot_both_succeed() {
    let root = test_root("concurrent-same-owner");
    let db = Database::open(root.clone()).unwrap();
    let adj = adjacency();
    let (first, second) = std::thread::scope(|scope| {
        let one = scope.spawn(|| adj.record_event(&db, "$e", &["$a"], &[], None));
        let two = scope.spawn(|| adj.record_event(&db, "$e", &["$b"], &[], None));
        (one.join().unwrap(), two.join().unwrap())
    });
    assert_eq!(
        usize::from(first.is_ok()) + usize::from(second.is_ok()),
        1,
        "two different prev sets for one event must not both win"
    );
    let error = first
        .err()
        .or_else(|| second.err())
        .expect("exactly one recording failed");
    assert!(matches!(error, StorageError::Collision(_)), "{error}");
    let stored = adj.prev_of(&db, "$e").unwrap().unwrap();
    assert!(
        stored == ["$a".to_owned()] || stored == ["$b".to_owned()],
        "the winner's edges must survive intact: {stored:?}"
    );
    assert!(adj.verify(&db).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn children_of_matches_the_derived_reverse_family() {
    let root = test_root("children-of");
    let db = Database::open(root.clone()).unwrap();
    let adj = adjacency();
    adj.record_event(&db, "$a", &[], &[], None).unwrap();
    // Fan-out in separate batches and a merge event with two parents.
    adj.record_event(&db, "$b", &["$a"], &[], None).unwrap();
    adj.record_event(&db, "$c", &["$a"], &[], None).unwrap();
    adj.record_events(
        &db,
        &[
            EventRecord {
                event_id: "$d",
                prev: &["$b", "$c"],
                auth: &[],
                relation: None,
            },
            EventRecord {
                event_id: "$e",
                prev: &["$a"],
                auth: &[],
                relation: None,
            },
        ],
    )
    .unwrap();
    // Re-recording identical data must not change the reverse rows.
    adj.record_event(&db, "$b", &["$a"], &[], None).unwrap();

    let derived = adj.reverse_family(&db, PREV).unwrap();
    for id in ["$a", "$b", "$c", "$d", "$e"] {
        let short = adj.short_id(&db, id).unwrap().unwrap();
        let expected: Option<Vec<u32>> = derived
            .children_of(short)
            .map(|edges| edges.iter().map(|edge| edge.target).collect());
        assert_eq!(adj.children_of(&db, short).unwrap(), expected, "{id}");
    }
    // The reverse rows must not break the forward snapshot or verification.
    assert_eq!(adj.verify(&db).unwrap().problems, Vec::<String>::new());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn proto_auth_children_merge_dedupe_and_survive_a_seal() {
    let root = test_root("proto-auth-children");
    let db = Database::open(root.clone()).unwrap();
    let adj = adjacency();
    let ids = adj
        .record_events(
            &db,
            &[
                EventRecord {
                    event_id: "$p",
                    prev: &[],
                    auth: &[],
                    relation: None,
                },
                EventRecord {
                    event_id: "$a",
                    prev: &[],
                    auth: &["$p"],
                    relation: None,
                },
                EventRecord {
                    event_id: "$b",
                    prev: &[],
                    auth: &["$p"],
                    relation: None,
                },
            ],
        )
        .unwrap();
    let (p, a, b) = (ids[0], ids[1], ids[2]);
    let sorted = |v: Option<Vec<u32>>| {
        let mut v = v.unwrap_or_default();
        v.sort_unstable();
        v
    };
    // A parent with no recorded child reads as absent.
    assert_eq!(adj.proto_auth_children_of(&db, p).unwrap(), None);
    adj.proto_maintain_auth_children(&db, &[(a, vec![p])])
        .unwrap();
    // A second batch merges with the stored child and repeats `a`.
    adj.proto_maintain_auth_children(&db, &[(a, vec![p]), (b, vec![p])])
        .unwrap();
    let mut expected = vec![a, b];
    expected.sort_unstable();
    assert_eq!(
        sorted(adj.proto_auth_children_of(&db, p).unwrap()),
        expected
    );
    assert_eq!(adj.proto_auth_children_of(&db, a).unwrap(), None);
    adj.proto_seal(&db).unwrap();
    assert_eq!(
        sorted(adj.proto_auth_children_of(&db, p).unwrap()),
        expected
    );
    assert_eq!(adj.proto_auth_children_of(&db, a).unwrap(), None);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn children_of_falls_back_then_backfills_a_legacy_scope() {
    let root = test_root("children-of-legacy");
    let db = Database::open(root.clone()).unwrap();
    let adj = adjacency();
    // Rows as a build without the reverse family wrote them.
    adj.record_legacy_prev(&db, &[("$b", "$a"), ("$c", "$a")])
        .unwrap();
    let a = adj.short_id(&db, "$a").unwrap().unwrap();
    let b = adj.short_id(&db, "$b").unwrap().unwrap();
    let c = adj.short_id(&db, "$c").unwrap().unwrap();
    // Unmarked: the derive fallback must still answer correctly.
    assert_eq!(adj.children_of(&db, a).unwrap(), Some(vec![b, c]));
    assert_eq!(adj.children_of(&db, b).unwrap(), None);
    // The next write backfills the old rows and marks the scope.
    adj.record_event(&db, "$d", &["$b"], &[], None).unwrap();
    let d = adj.short_id(&db, "$d").unwrap().unwrap();
    assert_eq!(adj.children_of(&db, a).unwrap(), Some(vec![b, c]));
    assert_eq!(adj.children_of(&db, b).unwrap(), Some(vec![d]));
    assert_eq!(adj.children_of(&db, d).unwrap(), None);
    assert_eq!(adj.verify(&db).unwrap().problems, Vec::<String>::new());
    let _ = std::fs::remove_dir_all(root);
}

/// Timing only: `cargo test --release --lib legacy_scope_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing run"]
fn legacy_scope_timing() {
    for events in [10_000_usize, 100_000] {
        let root = test_root("children-of-legacy-timing");
        let db = Database::open(root.clone()).unwrap();
        let adj = adjacency();
        // Spread ids (constant-prefix ids cluster the hash index) in a chain.
        let names: Vec<String> = (0..=events)
            .map(|n| format!("${:016x}", (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
            .collect();
        for chunk in names.windows(2).collect::<Vec<_>>().chunks(256) {
            let pairs: Vec<(&str, &str)> = chunk
                .iter()
                .map(|w| (w[1].as_str(), w[0].as_str()))
                .collect();
            adj.record_legacy_prev(&db, &pairs).unwrap();
        }
        let mid = adj.short_id(&db, &names[events / 2]).unwrap().unwrap();
        let t = std::time::Instant::now();
        let fallback = adj.children_of(&db, mid).unwrap();
        let fallback_time = t.elapsed();
        assert!(fallback.is_some());
        let t = std::time::Instant::now();
        adj.record_event(&db, "$new", &[names[events].as_str()], &[], None)
            .unwrap();
        let backfill_time = t.elapsed();
        let t = std::time::Instant::now();
        let fast = adj.children_of(&db, mid).unwrap();
        let fast_time = t.elapsed();
        assert_eq!(fast, fallback);
        println!(
            "legacy {events:>7} events: fallback children_of {fallback_time:?} | first write (backfill) {backfill_time:?} | children_of after {fast_time:?} | sealed segments {}",
            count_segments(&root)
        );
        let _ = std::fs::remove_dir_all(root);
    }
}

/// Times the `AUTH` reverse view and a rejection cascade over it.
///
/// Run: `cargo test --release --lib auth_cascade_timing -- --ignored --nocapture`
#[test]
#[ignore = "timing measurement, not a correctness test"]
fn auth_cascade_timing() {
    for events in [10_000_usize, 100_000] {
        let root = test_root("auth-cascade-timing");
        let db = Database::open(root.clone()).unwrap();
        let adj = adjacency();
        let names: Vec<String> = (0..=events)
            .map(|n| format!("${:016x}", (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
            .collect();
        // Event n cites the create event (0) and event n-1: a worst-case chain.
        let auths: Vec<[&str; 2]> = (1..=events)
            .map(|n| [names[0].as_str(), names[n - 1].as_str()])
            .collect();
        let records: Vec<EventRecord<'_>> = (1..=events)
            .map(|n| EventRecord {
                event_id: names[n].as_str(),
                prev: &[],
                auth: &auths[n - 1],
                relation: None,
            })
            .collect();
        for chunk in records.chunks(256) {
            adj.record_events(&db, chunk).unwrap();
        }
        let t = std::time::Instant::now();
        let reverse = adj.reverse_family(&db, AUTH).unwrap();
        let derive = t.elapsed();
        let cascade = |start: usize| {
            let t = std::time::Instant::now();
            let first = adj.short_id(&db, &names[start]).unwrap().unwrap();
            let mut seen = std::collections::HashSet::from([first]);
            let mut stack = vec![first];
            while let Some(parent) = stack.pop() {
                for edge in reverse.children_of(parent).unwrap_or(&[]) {
                    if seen.insert(edge.target) {
                        stack.push(edge.target);
                    }
                }
            }
            (seen.len() - 1, t.elapsed())
        };
        let (deep, deep_time) = cascade(events / 10);
        let (leaf, leaf_time) = cascade(events - 1);
        println!(
            "auth {events:>7} events: derive {derive:?} | cascade from 10%: {deep} events in {deep_time:?} | cascade from tip: {leaf} events in {leaf_time:?} | sealed segments {}",
            count_segments(&root)
        );
        let _ = std::fs::remove_dir_all(root);
    }
}

/// Throwaway benchmark: a maintained reverse `AUTH` family against deriving it,
/// on a star-shaped auth graph (every event cites create, power levels and one
/// of 100 member events). The rejection check is deliberately not measured.
///
/// Run: `cargo test --release --lib auth_star_timing -- --ignored --nocapture`
#[test]
#[ignore = "timing measurement, not a correctness test"]
#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::arithmetic_side_effects
)] // throwaway benchmark: readability over lint-cleanliness
fn auth_star_timing() {
    use std::time::{Duration, Instant};
    const MEMBERS: usize = 100;
    const BATCH: usize = 256;
    // AUTH_STAR_BASELINE=1 skips the prototype to measure the store's size without it.
    let baseline = std::env::var_os("AUTH_STAR_BASELINE").is_some();
    let avg = |total: Duration, n: usize| total / u32::try_from(n.max(1)).unwrap();
    for events in [10_000_usize, 100_000] {
        let root = test_root("auth-star-timing");
        let db = Database::open(root.clone()).unwrap();
        let adj = CsrAdjacency::new("!star:example.org");
        let name = |n: usize| format!("${:016x}", (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        // 0 = create, 1 = power levels, 2..2+MEMBERS = members, rest = messages.
        let first = 2 + MEMBERS;
        let names: Vec<String> = (0..first + events).map(name).collect();
        let auth_of = |n: usize| -> Vec<&str> {
            if n == 0 {
                vec![]
            } else if n < first {
                vec![names[0].as_str()]
            } else {
                vec![
                    names[0].as_str(),
                    names[1].as_str(),
                    names[2 + n % MEMBERS].as_str(),
                ]
            }
        };
        let mut write_core = Duration::ZERO;
        let mut write_proto = Duration::ZERO;
        let mut window = (0_usize, Duration::ZERO, Duration::ZERO);
        let mut short: Vec<u32> = Vec::new();
        let mut written = 0;
        println!("auth star, {events} events (per-event write cost as the create row grows):");
        while written < first + events {
            let end = (written + BATCH).min(first + events);
            let auths: Vec<Vec<&str>> = (written..end).map(auth_of).collect();
            let records: Vec<EventRecord<'_>> = (written..end)
                .map(|n| EventRecord {
                    event_id: names[n].as_str(),
                    prev: &[],
                    auth: &auths[n - written],
                    relation: None,
                })
                .collect();
            let t = Instant::now();
            let ids = adj.record_events(&db, &records).unwrap();
            let core = t.elapsed();
            short.extend(&ids);
            let batch: Vec<(u32, Vec<u32>)> = (written..end)
                .map(|n| {
                    let parents = if n == 0 {
                        Vec::new()
                    } else if n < first {
                        vec![short[0]]
                    } else {
                        vec![short[0], short[1], short[2 + n % MEMBERS]]
                    };
                    (short[n], parents)
                })
                .collect();
            let t = Instant::now();
            if !baseline {
                adj.proto_maintain_auth_children(&db, &batch).unwrap();
            }
            let proto = t.elapsed();
            write_core += core;
            write_proto += proto;
            window.0 += end - written;
            window.1 += core;
            window.2 += proto;
            written = end;
            if written % (events / 4 / BATCH * BATCH).max(BATCH) < BATCH && window.0 >= events / 4 {
                println!(
                    "  at {written:>7} events: record {:?}/event | maintain AUTH reverse {:?}/event",
                    avg(window.1, window.0),
                    avg(window.2, window.0)
                );
                window = (0, Duration::ZERO, Duration::ZERO);
            }
        }
        println!(
            "  whole build: record {:?}/event | maintain {:?}/event",
            avg(write_core, events),
            avg(write_proto, events)
        );
        println!(
            "  store after build: {} segments, {:.1} MiB on disk",
            count_segments(&root),
            dir_bytes(&root) as f64 / (1024.0 * 1024.0)
        );
        if baseline {
            let _ = std::fs::remove_dir_all(root);
            continue;
        }
        // Single-event writes against the fully grown create row.
        let (mut single_core, mut single_proto) = (Duration::ZERO, Duration::ZERO);
        for k in 0..100 {
            let n = first + events + k;
            let id = name(n);
            let auth = [names[0].as_str(), names[1].as_str(), names[2].as_str()];
            let t = Instant::now();
            let ids = adj
                .record_events(
                    &db,
                    &[EventRecord {
                        event_id: &id,
                        prev: &[],
                        auth: &auth,
                        relation: None,
                    }],
                )
                .unwrap();
            single_core += t.elapsed();
            let t = Instant::now();
            adj.proto_maintain_auth_children(&db, &[(ids[0], vec![short[0], short[1], short[2]])])
                .unwrap();
            single_proto += t.elapsed();
        }
        println!(
            "  single-event writes on a {events}-child create row: record {:?} | maintain {:?}",
            avg(single_core, 100),
            avg(single_proto, 100)
        );

        let reads = |label: &str| {
            let t = Instant::now();
            let view = adj.reverse_family(&db, AUTH).unwrap();
            let derive = t.elapsed();
            let probes = [
                ("create", short[0]),
                ("member", short[2]),
                ("message", short[first + events / 2]),
            ];
            for (what, id) in probes {
                let derived: Option<Vec<u32>> = view
                    .children_of(id)
                    .map(|e| e.iter().map(|edge| edge.target).collect());
                let t = Instant::now();
                let mut got = None;
                for _ in 0..20 {
                    got = adj.proto_auth_children_of(&db, id).unwrap();
                }
                let point = avg(t.elapsed(), 20);
                let mut a = derived.clone().unwrap_or_default();
                let mut b = got.clone().unwrap_or_default();
                a.sort_unstable();
                b.sort_unstable();
                // The 100 single-event writes after the build are not in `view`
                // only if it was derived earlier; both are read after them.
                assert_eq!(a, b, "{what} children differ");
                println!(
                    "  {label}: {what:<7} ({} children) maintained point read {point:?}",
                    a.len()
                );
            }
            println!("  {label}: derive reverse_family(AUTH) {derive:?}");
        };
        reads("tail only");
        let before = count_segments(&root);
        adj.proto_seal(&db).unwrap();
        println!(
            "  sealed segments: {before} before seal, {} after",
            count_segments(&root)
        );
        reads("sealed   ");
        let _ = std::fs::remove_dir_all(root);
    }
}

/// Number of sealed `.csr` segments under `dir` (the store seals its tail at
/// 4 MiB on its own, so "tail only" holds only for small runs).
#[allow(clippy::arithmetic_side_effects)] // benchmark helper
fn count_segments(dir: &std::path::Path) -> usize {
    let mut n = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|e| e == "csr") {
                n += 1;
            }
        }
    }
    n
}

/// Total bytes of all files under `dir`.
#[allow(clippy::arithmetic_side_effects)] // benchmark helper
fn dir_bytes(dir: &std::path::Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if let Ok(meta) = p.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

#[test]
fn for_collection_scopes_by_collection_id_and_serves_children_of() {
    let root = test_root("for-collection");
    let db = Database::open(root.clone()).unwrap();
    let room_a = [0xa1; 16];
    let room_b = [0xb2; 16];
    let a = CsrAdjacency::for_collection(&room_a);
    a.record_event(&db, "$create", &[], &[], None).unwrap();
    a.record_event(&db, "$msg", &["$create"], &["$create"], None)
        .unwrap();

    // A second handle on the same collection id reaches the same rows.
    let again = CsrAdjacency::for_collection(&room_a);
    assert_eq!(
        again.prev_of(&db, "$msg").unwrap(),
        Some(vec!["$create".to_owned()])
    );
    let create = again.short_id(&db, "$create").unwrap().unwrap();
    let children = again.children_of(&db, create).unwrap().unwrap();
    assert_eq!(
        again.resolve_event_ids(&db, &children).unwrap(),
        vec![Some("$msg".to_owned())]
    );

    // Another collection id does not see them.
    let other = CsrAdjacency::for_collection(&room_b);
    assert_eq!(other.prev_of(&db, "$msg").unwrap(), None);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[cfg(feature = "multi-reader")]
#[test]
fn read_only_graph_reads_what_the_writer_committed() {
    use crate::layout::{DatabaseLayout, ShardType};
    use crate::packfile::storage::PackfileStorage;
    use crate::short_id::ReadOnlyGraph;
    use std::sync::Arc;

    let root = test_root("read-only-graph");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0xc3; 16];
    let adj = CsrAdjacency::for_collection(&collection);
    adj.record_event(&db, "$create", &[], &[], None).unwrap();
    adj.record_event(&db, "$msg", &["$create"], &["$create"], None)
        .unwrap();

    let layout = DatabaseLayout::open(root.clone()).unwrap();
    let pools: Vec<Arc<PackfileStorage>> = ShardType::ALL
        .iter()
        .map(|&pool| {
            Arc::new(
                PackfileStorage::open_read_committed_shared(
                    layout.pool_dir(pool).unwrap(),
                    layout.shared_wal_path(),
                    pool,
                )
                .unwrap(),
            )
        })
        .collect();
    let reader = ReadOnlyGraph::new(layout, pools.try_into().ok().unwrap());

    assert_eq!(
        adj.prev_of(&reader, "$msg").unwrap(),
        Some(vec!["$create".to_owned()])
    );
    let create = adj.short_id(&reader, "$create").unwrap().unwrap();
    let children = adj.children_of(&reader, create).unwrap().unwrap();
    assert_eq!(
        adj.resolve_event_ids(&reader, &children).unwrap(),
        vec![Some("$msg".to_owned())]
    );
    drop(reader);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn durable_record_events_syncs_once_outside_the_write_lock() {
    let root = test_root("sync-outside-lock");
    let db = Database::open(root.clone()).unwrap();
    let adj = adjacency();
    let before = crate::csr::store::test_tail_syncs();
    adj.record_event(&db, "$create", &[], &[], None).unwrap();
    assert_eq!(
        crate::csr::store::test_tail_syncs() - before,
        1,
        "one durable recording is one tail sync"
    );
    assert!(
        !crate::csr::store::test_last_sync_held_lock(),
        "the sync must run after the store write lock is released"
    );
    let before_noop = crate::csr::store::test_tail_syncs();
    adj.record_event(&db, "$create", &[], &[], None).unwrap();
    assert_eq!(
        crate::csr::store::test_tail_syncs() - before_noop,
        1,
        "a durable no-op still establishes the durability barrier"
    );
    let before_empty = crate::csr::store::test_tail_syncs();
    adj.record_events_with(&db, &[], AppendDurability::Durable)
        .unwrap();
    assert_eq!(
        crate::csr::store::test_tail_syncs() - before_empty,
        1,
        "an empty durable batch still establishes the durability barrier"
    );
    let _ = std::fs::remove_dir_all(root);
}

// ---- edge log: journaled edges, rebuilt into the CSR after a crash ----

mod edge_log_recovery {
    use super::*;
    use crate::csr::store::{CsrStore, RowRecord, ScopeId};
    use crate::edge_log::{EdgeLog, LoggedEvent};
    use crate::layout::ShardType;

    const ROOM_A: [u8; 16] = [0xA1; 16];
    const ROOM_B: [u8; 16] = [0xB2; 16];

    fn event<'a>(id: &'a str, prev: &'a [&'a str]) -> EventRecord<'a> {
        EventRecord {
            event_id: id,
            prev,
            auth: &["$create"],
            relation: None,
        }
    }

    /// Make the WAL durable (the log with it) but lose the unsynced CSR tail,
    /// then reopen: the state a crash leaves behind.
    fn crash_keeping_only_the_wal(root: &std::path::Path, db: Database) -> Database {
        db.coordinator().sync().unwrap();
        let tail = db.layout().graph_path(ShardType::State).join("tail.log");
        drop(db);
        let _ = std::fs::remove_file(tail);
        Database::open(root.to_path_buf()).unwrap()
    }

    /// A restart after the tail was already synced: nothing is lost.
    fn restart_keeping_everything(root: &std::path::Path, db: Database) -> Database {
        db.coordinator().sync().unwrap();
        drop(db);
        Database::open(root.to_path_buf()).unwrap()
    }

    fn scope_with_applied_purge(store: &CsrStore) -> ScopeId {
        store
            .all_scopes()
            .unwrap()
            .into_iter()
            .map(ScopeId)
            .find(|scope| {
                store
                    .read_scope(*scope)
                    .unwrap()
                    .rows
                    .iter()
                    .any(|row| row.family.id == 41)
            })
            .expect("purge scope has an applied marker")
    }

    #[test]
    fn rollback_repair_preserves_scopes_without_a_recovered_counter() {
        let root = test_root("edge-log-unknown-scope");
        let db = Database::open(root.clone()).unwrap();
        let store = CsrStore::open(db.layout(), ShardType::State);
        let scope = ScopeId([0xD4; 16]);
        let row = RowRecord {
            family: PREV,
            owner: 1,
            edges: vec![crate::csr::Edge { target: 2, kind: 0 }],
        };
        store
            .writer(&db)
            .unwrap()
            .append_rows(scope, std::slice::from_ref(&row))
            .unwrap();

        CsrAdjacency::recover_edge_log(&db, &EdgeLog::new()).unwrap();

        assert_eq!(store.read_scope(scope).unwrap().rows.len(), 1);
        assert_eq!(store.read_scope(scope).unwrap().rows[0].edges, row.edges);
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn clean_recovery_syncs_the_existing_tail_once() {
        let root = test_root("edge-log-clean-recovery-sync-count");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        CsrAdjacency::for_collection(&ROOM_A)
            .record_events_logged(
                &db,
                &log,
                &[event("$clean-recovery", &[])],
                AppendDurability::Durable,
            )
            .unwrap();
        let db = restart_keeping_everything(&root, db);

        let before = crate::csr::store::test_tail_syncs();
        assert_eq!(CsrAdjacency::recover_edge_log(&db, &log).unwrap(), 0);
        assert_eq!(
            crate::csr::store::test_tail_syncs() - before,
            1,
            "clean recovery needs its initial durability barrier but no rewrite barrier"
        );

        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_missing_entry_above_the_prune_point_is_corruption() {
        let root = test_root("edge-log-gap");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        for (id, prev) in [("$a", &[][..]), ("$b", &["$a"][..]), ("$c", &["$b"][..])] {
            adj.record_events_logged(&db, &log, &[event(id, prev)], AppendDurability::Deferred)
                .unwrap();
        }
        // Lose entry 2 from the journal, as a damaged log would.
        log.drop_entry_for_test(&db, 2);
        let db = crash_keeping_only_the_wal(&root, db);
        let err = CsrAdjacency::recover_edge_log(&db, &log).unwrap_err();
        assert!(err.to_string().contains("missing entries"), "{err}");
    }

    #[test]
    fn a_lost_tail_watermark_can_still_skip_the_pruned_prefix() {
        let root = test_root("edge-log-pruned-watermark");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        for (id, prev) in [("$a", &[][..]), ("$b", &["$a"][..])] {
            adj.record_events_logged(&db, &log, &[event(id, prev)], AppendDurability::Deferred)
                .unwrap();
        }
        adj.sync_tail(&db).unwrap();
        CsrAdjacency::checkpoint_edge_log(&db, &log).unwrap();
        assert_eq!(log.pruned(&db).unwrap(), 2);
        // Keep the covered prefix in an immutable segment; the crash helper
        // removes only the tail, where the watermark and later suffix live.
        adj.proto_seal(&db).unwrap();

        for (id, prev) in [("$c", &["$b"][..]), ("$d", &["$c"][..])] {
            adj.record_events_logged(&db, &log, &[event(id, prev)], AppendDurability::Deferred)
                .unwrap();
        }

        // Lose the whole tail, watermark included; the WAL keeps the log and
        // the prune marker. Entries 1-2 are legitimately gone, so recovery
        // must look past the pruned prefix instead of reporting a gap.
        let db = crash_keeping_only_the_wal(&root, db);
        assert_eq!(log.head(&db).unwrap(), 4);
        assert_eq!(log.pruned(&db).unwrap(), 2);
        assert_eq!(
            CsrAdjacency::recover_edge_log(&db, &log).unwrap(),
            2,
            "only the entries above the prune point are replayed"
        );
        assert_eq!(adj.prev_of(&db, "$b").unwrap(), Some(vec!["$a".to_owned()]));
        assert_eq!(adj.prev_of(&db, "$d").unwrap(), Some(vec!["$c".to_owned()]));
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn the_watermark_never_passes_an_unapplied_entry() {
        let root = test_root("edge-log-gap-watermark");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        // Entry 1 was logged but its rows never reached the tail, as after a
        // failed append.
        let lost = LoggedEvent {
            scope: ROOM_B,
            event_id: "$lost".to_owned(),
            prev: Vec::new(),
            auth: Vec::new(),
            relation: None,
        };
        assert_eq!(log.append(&db, &[lost]).unwrap(), 1);
        // A later successful batch must not advance the watermark over it.
        CsrAdjacency::for_collection(&ROOM_A)
            .record_events_logged(&db, &log, &[event("$a", &[])], AppendDurability::Deferred)
            .unwrap();
        assert_eq!(
            CsrAdjacency::recover_edge_log(&db, &log).unwrap(),
            2,
            "recovery replays from before the gap"
        );
        assert_eq!(
            CsrAdjacency::for_collection(&ROOM_B)
                .prev_of(&db, "$lost")
                .unwrap(),
            Some(Vec::new()),
            "the unapplied entry's rows were rebuilt"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn recovery_does_not_resurrect_rows_from_before_a_scope_purge() {
        let root = test_root("edge-log-purge-boundary");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        for (id, prev) in [("$a", &[][..]), ("$b", &["$a"][..])] {
            adj.record_events_logged(&db, &log, &[event(id, prev)], AppendDurability::Deferred)
                .unwrap();
        }

        adj.purge(&db).unwrap();
        adj.record_events_logged(
            &db,
            &log,
            &[event("$after-purge", &[])],
            AppendDurability::Deferred,
        )
        .unwrap();
        let entries = log.read_range(&db, 0, log.head(&db).unwrap()).unwrap();
        assert_eq!(
            entries
                .iter()
                .filter(|(_, event)| EdgeLog::is_scope_purge(event))
                .count(),
            1,
            "purge is represented in the edge log"
        );

        let db = crash_keeping_only_the_wal(&root, db);
        assert_eq!(
            CsrAdjacency::recover_edge_log(&db, &log).unwrap(),
            1,
            "only the event recorded after the purge is replayed"
        );
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        assert_eq!(adj.short_id(&db, "$a").unwrap(), None);
        assert_eq!(adj.short_id(&db, "$b").unwrap(), None);
        assert_eq!(adj.prev_of(&db, "$after-purge").unwrap(), Some(Vec::new()));
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn applied_purge_preserves_later_unlogged_rows_across_a_log_gap() {
        use crate::database::tests::crash_image;

        let root = test_root("edge-log-purge-applied-gap");
        let image = test_root("edge-log-purge-applied-gap-image");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        adj.record_event(&db, "$before", &[], &[], None).unwrap();
        assert_eq!(adj.prev_of(&db, "$before").unwrap(), Some(Vec::new()));

        log.append(
            &db,
            &[LoggedEvent {
                scope: ROOM_B,
                event_id: "$unapplied".to_owned(),
                prev: Vec::new(),
                auth: Vec::new(),
                relation: None,
            }],
        )
        .unwrap();
        adj.purge_graph(&db).unwrap();
        adj.record_event(&db, "$after", &[], &[], None).unwrap();
        adj.sync_tail(&db).unwrap();
        db.coordinator().sync().unwrap();

        crash_image(&db, &root, &image);
        drop(db);
        let db = Database::open(image.clone()).unwrap();
        CsrAdjacency::recover_edge_log(&db, &log).unwrap();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        assert_eq!(adj.prev_of(&db, "$before").unwrap(), None);
        assert_eq!(adj.prev_of(&db, "$after").unwrap(), Some(Vec::new()));
        drop(db);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(image);
    }

    #[test]
    fn recovery_applies_a_purge_marker_when_its_tombstone_was_not_written() {
        use crate::database::tests::crash_image;

        let root = test_root("edge-log-purge-marker-only");
        let image = test_root("edge-log-purge-marker-only-image");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        adj.record_event(&db, "$before", &[], &[], None).unwrap();
        adj.sync_tail(&db).unwrap();
        db.coordinator().sync().unwrap();
        log.append_scope_purge(&db, ROOM_A).unwrap();
        db.coordinator().sync().unwrap();

        crash_image(&db, &root, &image);
        drop(db);
        let db = Database::open(image.clone()).unwrap();
        CsrAdjacency::recover_edge_log(&db, &log).unwrap();
        assert_eq!(adj.prev_of(&db, "$before").unwrap(), None);
        drop(db);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(image);
    }

    #[test]
    fn recovery_applies_only_the_latest_purge_for_a_scope() {
        use crate::database::tests::crash_image;

        let root = test_root("edge-log-latest-purge");
        let image = test_root("edge-log-latest-purge-image");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        log.append(
            &db,
            &[LoggedEvent {
                scope: ROOM_B,
                event_id: "$unapplied".to_owned(),
                prev: Vec::new(),
                auth: Vec::new(),
                relation: None,
            }],
        )
        .unwrap();
        adj.record_event(&db, "$before", &[], &[], None).unwrap();
        adj.purge_graph(&db).unwrap();
        adj.record_event(&db, "$between", &[], &[], None).unwrap();
        adj.purge_graph(&db).unwrap();
        adj.record_event(&db, "$after", &[], &[], None).unwrap();
        adj.sync_tail(&db).unwrap();
        db.coordinator().sync().unwrap();

        crash_image(&db, &root, &image);
        drop(db);
        let db = Database::open(image.clone()).unwrap();
        CsrAdjacency::recover_edge_log(&db, &log).unwrap();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        assert_eq!(adj.prev_of(&db, "$before").unwrap(), None);
        assert_eq!(adj.prev_of(&db, "$between").unwrap(), None);
        assert_eq!(adj.prev_of(&db, "$after").unwrap(), Some(Vec::new()));
        drop(db);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(image);
    }

    #[test]
    fn a_reused_purge_sequence_with_a_new_token_is_applied() {
        use crate::database::tests::crash_image;
        use crate::journal::cut_segment_image_to_durable_mark;

        let root = test_root("edge-log-purge-token-reuse");
        let image = test_root("edge-log-purge-token-reuse-image");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        let owner = adj.record_event(&db, "$old", &[], &[], None).unwrap();
        log.append(
            &db,
            &[LoggedEvent {
                scope: ROOM_B,
                event_id: "$durable".to_owned(),
                prev: Vec::new(),
                auth: Vec::new(),
                relation: None,
            }],
        )
        .unwrap();
        db.coordinator().sync().unwrap();
        let before_purge = log.head(&db).unwrap();
        assert_eq!(before_purge, 1);
        adj.purge_graph(&db).unwrap();
        let (first_seq, first_marker) = log
            .read_range(&db, 0, log.head(&db).unwrap())
            .unwrap()
            .into_iter()
            .find(|(_, entry)| EdgeLog::is_scope_purge(entry))
            .expect("purge marker was logged");
        let first_token = EdgeLog::purge_token(&first_marker).unwrap();
        assert_eq!(first_seq, 2);

        let store = CsrStore::open(db.layout(), ShardType::State);
        let scope = scope_with_applied_purge(&store);
        store
            .writer(&db)
            .unwrap()
            .append_rows(
                scope,
                &[RowRecord {
                    family: PREV,
                    owner,
                    edges: vec![crate::csr::Edge {
                        target: owner,
                        kind: 0,
                    }],
                }],
            )
            .unwrap();
        assert_eq!(
            adj.prev_of(&db, "$old").unwrap(),
            Some(vec!["$old".to_owned()])
        );

        crash_image(&db, &root, &image);
        cut_segment_image_to_durable_mark(&image.join("wal.bin")).unwrap();
        drop(db);
        let db = Database::open(image.clone()).unwrap();
        assert_eq!(log.head(&db).unwrap(), 1);
        let (new_seq, new_token) = log.append_scope_purge(&db, ROOM_A).unwrap();
        assert_eq!(new_seq, first_seq);
        assert_ne!(new_token, first_token);
        CsrAdjacency::recover_edge_log(&db, &log).unwrap();
        assert_eq!(adj.prev_of(&db, "$old").unwrap(), None);
        drop(db);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(image);
    }

    #[test]
    fn an_applied_purge_past_the_log_head_stays_a_floor_for_reused_numbers() {
        use crate::database::tests::crash_image;
        use crate::journal::cut_segment_image_to_durable_mark;

        let root = test_root("edge-log-purge-floor");
        let image = test_root("edge-log-purge-floor-image");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        // The gap keeps the watermark behind everything after it.
        leave_a_gap(&db, &log);
        adj.record_events_logged(&db, &log, &[event("$a", &[])], AppendDurability::Deferred)
            .unwrap();
        db.coordinator().sync().unwrap();
        // The purge (log sequence 3) reaches the tail, but its WAL group does not.
        adj.purge_graph(&db).unwrap();

        crash_image(&db, &root, &image);
        cut_segment_image_to_durable_mark(&image.join("wal.bin")).unwrap();
        drop(db);
        let db = Database::open(image.clone()).unwrap();
        assert_eq!(log.head(&db).unwrap(), 2, "the purge entry was rolled back");

        CsrAdjacency::recover_edge_log(&db, &log).unwrap();
        assert_eq!(
            adj.prev_of(&db, "$a").unwrap(),
            None,
            "the purge applied at sequence 3 still hides the earlier entry"
        );
        assert_eq!(
            log.head(&db).unwrap(),
            3,
            "numbers up to the applied purge are not handed out again"
        );

        // The next entry gets a number above the purge, so a later recovery
        // cannot mistake it for one that precedes the purge.
        adj.record_events_logged(&db, &log, &[event("$new", &[])], AppendDurability::Deferred)
            .unwrap();
        assert_eq!(log.head(&db).unwrap(), 4);
        // Copy the store as a crash would leave it, then check the live one.
        let later = test_root("edge-log-purge-floor-later");
        db.coordinator().sync().unwrap();
        crash_image(&db, &image, &later);
        // The watermark moved with the floor, so the new entry can be pruned.
        CsrAdjacency::checkpoint_edge_log(&db, &log).unwrap();
        assert_eq!(log.pruned(&db).unwrap(), 4);
        // A normal restart after the checkpoint keeps the pruned entry's rows.
        let db = restart_keeping_everything(&image, db);
        assert_eq!(CsrAdjacency::recover_edge_log(&db, &log).unwrap(), 0);
        assert_eq!(adj.prev_of(&db, "$new").unwrap(), Some(Vec::new()));
        assert_eq!(adj.prev_of(&db, "$a").unwrap(), None);
        drop(db);
        let db = Database::open(later.clone()).unwrap();
        let db = crash_keeping_only_the_wal(&later, db);
        CsrAdjacency::recover_edge_log(&db, &log).unwrap();
        assert_eq!(adj.prev_of(&db, "$new").unwrap(), Some(Vec::new()));
        assert_eq!(adj.prev_of(&db, "$a").unwrap(), None);
        drop(db);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(image);
        let _ = std::fs::remove_dir_all(later);
    }

    #[test]
    fn a_scope_with_only_an_applied_purge_still_sets_the_floor() {
        use crate::database::tests::crash_image;
        use crate::journal::cut_segment_image_to_durable_mark;

        let root = test_root("edge-log-purge-floor-marker-only");
        let image = test_root("edge-log-purge-floor-marker-only-image");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        // No events and no log entries for this scope, only its purge.
        CsrAdjacency::for_collection(&ROOM_B)
            .purge_graph(&db)
            .unwrap();

        crash_image(&db, &root, &image);
        cut_segment_image_to_durable_mark(&image.join("wal.bin")).unwrap();
        drop(db);
        let db = Database::open(image.clone()).unwrap();
        assert_eq!(log.head(&db).unwrap(), 0);

        CsrAdjacency::recover_edge_log(&db, &log).unwrap();

        assert_eq!(log.head(&db).unwrap(), 1);
        assert_eq!(log.pruned(&db).unwrap(), 1);
        drop(db);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(image);
    }

    #[test]
    fn repair_drops_an_orphan_row_even_when_the_watermark_equals_the_head() {
        let root = test_root("edge-log-equal-head-orphan");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        adj.record_events_logged(&db, &log, &[event("$a", &[])], AppendDurability::Deferred)
            .unwrap();
        adj.sync_tail(&db).unwrap();

        // A row for an owner the recovered counter never allocated, as left by
        // an allocation that rolled back while the log entry survived.
        let store = CsrStore::open(db.layout(), ShardType::State);
        let scope = store
            .all_scopes()
            .unwrap()
            .into_iter()
            .map(ScopeId)
            .find(|scope| !store.read_scope(*scope).unwrap().rows.is_empty())
            .expect("the room has rows");
        let orphan_owner = 99;
        store
            .writer(&db)
            .unwrap()
            .append_rows(
                scope,
                &[RowRecord {
                    family: PREV,
                    owner: orphan_owner,
                    edges: Vec::new(),
                }],
            )
            .unwrap();
        let has_orphan = |store: &CsrStore| {
            store
                .read_scope(scope)
                .unwrap()
                .rows
                .iter()
                .any(|row| row.owner == orphan_owner)
        };
        assert!(has_orphan(&store));

        let db = restart_keeping_everything(&root, db);
        assert_eq!(
            log.head(&db).unwrap(),
            1,
            "the watermark equals the head, so nothing replays"
        );
        let before = crate::csr::store::test_tail_syncs();
        assert_eq!(CsrAdjacency::recover_edge_log(&db, &log).unwrap(), 0);
        assert_eq!(
            crate::csr::store::test_tail_syncs() - before,
            2,
            "a repair that changes rows must sync them after the initial barrier"
        );

        let store = CsrStore::open(db.layout(), ShardType::State);
        assert!(!has_orphan(&store), "the orphan row must be repaired away");
        assert_eq!(adj.prev_of(&db, "$a").unwrap(), Some(Vec::new()));
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    /// A room's rows as `(family, owner, sorted edges)`, for set comparison
    /// across a crash and reopen.
    type Links = Vec<(u16, u32, Vec<(u32, u16)>)>;

    fn scope_links(store: &CsrStore, scope: ScopeId) -> Links {
        let mut rows: Links = store
            .read_scope(scope)
            .unwrap()
            .rows
            .into_iter()
            .map(|row| {
                let mut edges: Vec<(u32, u16)> = row
                    .edges
                    .iter()
                    .map(|edge| (edge.target, edge.kind))
                    .collect();
                edges.sort_unstable();
                (row.family.id, row.owner, edges)
            })
            .collect();
        rows.sort_unstable();
        rows
    }

    /// The crash test's room: a chain with one relation, so prev, auth and
    /// typed relation rows are all covered.
    fn crash_events() -> [EventRecord<'static>; 5] {
        [
            event("$a", &[]),
            event("$b", &["$a"]),
            event("$c", &["$b"]),
            event("$d", &["$c"]),
            EventRecord {
                event_id: "$r",
                prev: &["$d"],
                auth: &["$create"],
                relation: Some(RelationRef {
                    target: "$d",
                    rel_type: "m.annotation",
                }),
            },
        ]
    }

    /// The edge log is pruned before the repair, so the CSR store holds the
    /// only copy of these links. A crash around the repair must not lose any.
    #[test]
    fn a_crash_around_a_repair_keeps_every_link() {
        let root = test_root("edge-log-repair-crash");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        adj.record_events_logged(&db, &log, &crash_events(), AppendDurability::Durable)
            .unwrap();
        CsrAdjacency::checkpoint_edge_log(&db, &log).unwrap();
        assert_eq!(log.pruned(&db).unwrap(), log.head(&db).unwrap());

        let store = CsrStore::open(db.layout(), ShardType::State);
        let scope = store
            .all_scopes()
            .unwrap()
            .into_iter()
            .map(ScopeId)
            .find(|scope| !store.read_scope(*scope).unwrap().rows.is_empty())
            .expect("the room has rows");
        let expected = scope_links(&store, scope);
        assert!(expected.iter().any(|(family, _, edges)| {
            *family == RELATIONS.id && edges.iter().any(|(_, kind)| *kind > 0)
        }));
        // An owner the recovered counter never allocated makes repair run.
        let orphan = RowRecord {
            family: PREV,
            owner: 99,
            edges: Vec::new(),
        };
        store
            .writer(&db)
            .unwrap()
            .append_rows(scope, &[orphan])
            .unwrap();
        let dir = db.layout().graph_path(ShardType::State);
        let tail = dir.join("tail.log");
        let old_tail = std::fs::read(&tail).unwrap();

        // Crash while writing the new segment: only a partial temp file exists.
        std::fs::write(dir.join(".partial.tmp"), b"partial").unwrap();
        let db = restart_keeping_everything(&root, db);
        CsrAdjacency::recover_edge_log(&db, &log).unwrap();
        let store = CsrStore::open(db.layout(), ShardType::State);
        assert_eq!(scope_links(&store, scope), expected);
        assert_eq!(
            std::fs::metadata(&tail).unwrap().len(),
            0,
            "repair must publish a segment, not rewrite rows in the tail"
        );

        // Crash after the segment is published, with the tail truncate lost:
        // the old tail overrides the segment until repair runs again.
        drop(db);
        std::fs::write(&tail, &old_tail).unwrap();
        let db = Database::open(root.clone()).unwrap();
        CsrAdjacency::recover_edge_log(&db, &log).unwrap();
        let store = CsrStore::open(db.layout(), ShardType::State);
        assert_eq!(scope_links(&store, scope), expected);

        adj.record_events_logged(
            &db,
            &log,
            &[event("$e", &["$d"])],
            AppendDurability::Durable,
        )
        .unwrap();
        let db = restart_keeping_everything(&root, db);
        CsrAdjacency::recover_edge_log(&db, &log).unwrap();
        assert_eq!(adj.prev_of(&db, "$e").unwrap(), Some(vec!["$d".to_owned()]));
        let d = adj.short_id(&db, "$d").unwrap().unwrap();
        let e = adj.short_id(&db, "$e").unwrap().unwrap();
        assert!(
            adj.children_of(&db, d)
                .unwrap()
                .is_some_and(|children| children.contains(&e)),
            "$d must list the appended event as a child after reopen"
        );
        let store = CsrStore::open(db.layout(), ShardType::State);
        let after = scope_links(&store, scope);
        for (family, owner, edges) in &expected {
            assert!(
                after.iter().any(|(f, o, new)| f == family
                    && o == owner
                    && edges.iter().all(|edge| new.contains(edge))),
                "lost links in row family={family} owner={owner}"
            );
        }
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    /// A crash between publishing the replacement segment and truncating the
    /// tail is not old-or-new: the restored tail re-adds the rows the repair
    /// removed from the tail, while rows it removed from an older sealed
    /// segment stay gone. Recovery must converge that mixed state.
    #[test]
    fn a_lost_tail_truncate_mixes_sealed_and_tail_rows() {
        let root = test_root("edge-log-mixed-repair-window");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        adj.record_events_logged(&db, &log, &[event("$a", &[])], AppendDurability::Durable)
            .unwrap();
        adj.sync_tail(&db).unwrap();

        let store = CsrStore::open(db.layout(), ShardType::State);
        let scope = store
            .all_scopes()
            .unwrap()
            .into_iter()
            .map(ScopeId)
            .find(|scope| !store.read_scope(*scope).unwrap().rows.is_empty())
            .expect("the room has rows");
        let expected = scope_links(&store, scope);

        let orphan = |owner| RowRecord {
            family: PREV,
            owner,
            edges: Vec::new(),
        };
        // One orphan is sealed into an older segment, the other stays in the
        // tail. Both name ids the recovered counter never allocated.
        store
            .writer(&db)
            .unwrap()
            .append_rows(scope, &[orphan(99)])
            .unwrap();
        store.seal().unwrap();
        store
            .writer(&db)
            .unwrap()
            .append_rows(scope, &[orphan(98)])
            .unwrap();
        let tail = db.layout().graph_path(ShardType::State).join("tail.log");
        let old_tail = std::fs::read(&tail).unwrap();
        assert_eq!(
            scope_links(&store, scope).len(),
            expected.len() + 2,
            "both orphans must be visible before the repair"
        );

        // The first recovery drops both orphans and publishes a new segment.
        CsrAdjacency::recover_edge_log(&db, &log).unwrap();
        let store = CsrStore::open(db.layout(), ShardType::State);
        assert_eq!(scope_links(&store, scope), expected);

        // Recreate the tail a crashed truncate would leave behind, then open
        // without recovering: the tail rows are back, the sealed row is not.
        drop(db);
        std::fs::write(&tail, &old_tail).unwrap();
        let db = Database::open(root.clone()).unwrap();
        let store = CsrStore::open(db.layout(), ShardType::State);
        let mixed = scope_links(&store, scope);
        assert!(
            mixed
                .iter()
                .any(|(_, owner, edges)| *owner == 98 && edges.is_empty()),
            "a row deleted from the tail must reappear"
        );
        assert!(
            !mixed.iter().any(|(_, owner, _)| *owner == 99),
            "a row deleted from the sealed segment must stay repaired"
        );
        assert_ne!(mixed, expected, "the window must be mixed, not old-or-new");

        // Recovery runs the filter again and converges.
        CsrAdjacency::recover_edge_log(&db, &log).unwrap();
        let store = CsrStore::open(db.layout(), ShardType::State);
        assert_eq!(scope_links(&store, scope), expected);
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn rollback_repair_keeps_a_purge_marker_for_an_unmapped_scope() {
        use crate::database::tests::crash_image;

        let root = test_root("edge-log-purge-marker-unmapped");
        let image = test_root("edge-log-purge-marker-unmapped-image");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        log.append(
            &db,
            &[LoggedEvent {
                scope: ROOM_A,
                event_id: "$unapplied".to_owned(),
                prev: Vec::new(),
                auth: Vec::new(),
                relation: None,
            }],
        )
        .unwrap();
        let adj = CsrAdjacency::for_collection(&ROOM_B);
        adj.purge_graph(&db).unwrap();
        log.append(
            &db,
            &[LoggedEvent {
                scope: ROOM_B,
                event_id: "$after".to_owned(),
                prev: Vec::new(),
                auth: Vec::new(),
                relation: None,
            }],
        )
        .unwrap();
        db.coordinator().sync().unwrap();

        crash_image(&db, &root, &image);
        drop(db);
        let db = Database::open(image.clone()).unwrap();
        CsrAdjacency::recover_edge_log(&db, &log).unwrap();
        let store = CsrStore::open(db.layout(), ShardType::State);
        let scope = scope_with_applied_purge(&store);
        assert!(store
            .read_scope(scope)
            .unwrap()
            .rows
            .iter()
            .any(|row| row.family.id == 41));
        assert_eq!(adj.prev_of(&db, "$after").unwrap(), Some(Vec::new()));
        drop(db);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(image);
    }

    #[test]
    fn watermark_ahead_of_the_log_head_keeps_only_surviving_allocations() {
        use crate::database::tests::crash_image;
        use crate::journal::cut_segment_image_to_durable_mark;

        let root = test_root("edge-log-clamp");
        let image = test_root("edge-log-clamp-image");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();

        // Room A's log entries and short-id allocations are durable before the
        // tail is synced, so its rows must survive the repair.
        let kept = CsrAdjacency::for_collection(&ROOM_A);
        kept.record_events_logged(
            &db,
            &log,
            &[event("$a", &[]), event("$b", &["$a"])],
            AppendDurability::Deferred,
        )
        .unwrap();
        db.coordinator().sync().unwrap();

        // Room B's allocations are only in the WAL. Syncing the tail makes its
        // rows and the watermark durable while the covering WAL group is not.
        let lost = CsrAdjacency::for_collection(&ROOM_B);
        lost.record_events_logged(
            &db,
            &log,
            &[event("$lost", &[])],
            AppendDurability::Deferred,
        )
        .unwrap();
        lost.sync_tail(&db).unwrap();

        crash_image(&db, &root, &image);
        cut_segment_image_to_durable_mark(&image.join("wal.bin")).unwrap();
        drop(db);

        let db = Database::open(image.clone()).unwrap();
        assert_eq!(log.head(&db).unwrap(), 2, "only room A's entries survived");
        assert_eq!(CsrAdjacency::recover_edge_log(&db, &log).unwrap(), 0);
        assert_eq!(
            kept.prev_of(&db, "$b").unwrap(),
            Some(vec!["$a".to_owned()]),
            "rows covered by durable allocations are preserved"
        );
        assert_eq!(
            lost.prev_of(&db, "$lost").unwrap(),
            None,
            "rows whose short-id allocation rolled back are dropped"
        );

        // The repaired store still accepts and recovers new logged appends.
        lost.record_events_logged(
            &db,
            &log,
            &[event("$after", &[])],
            AppendDurability::Deferred,
        )
        .unwrap();
        let db = crash_keeping_only_the_wal(&image, db);
        assert_eq!(CsrAdjacency::recover_edge_log(&db, &log).unwrap(), 1);
        assert_eq!(
            CsrAdjacency::for_collection(&ROOM_B)
                .prev_of(&db, "$after")
                .unwrap(),
            Some(Vec::new())
        );
        drop(db);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(image);
    }

    #[test]
    fn watermark_repair_drops_relation_edges_with_rolled_back_kinds() {
        use crate::database::tests::crash_image;
        use crate::journal::cut_segment_image_to_durable_mark;
        use crate::storage::NodeData;

        let root = test_root("edge-log-clamp-relation-kind");
        let image = test_root("edge-log-clamp-relation-kind-image");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        let plain = |id| EventRecord {
            event_id: id,
            prev: &[],
            auth: &[],
            relation: None,
        };
        let related = |id, rel_type| EventRecord {
            event_id: id,
            prev: &[],
            auth: &[],
            relation: Some(RelationRef {
                target: "$a",
                rel_type,
            }),
        };
        // Repair finds a scope's kind counter through its room collection in
        // the event pool, so the room must exist before the durable boundary.
        let txn = db.begin_transaction();
        txn.put(
            ShardType::EventDag,
            ROOM_A,
            [1; 16],
            &NodeData::new(bytes::Bytes::from_static(b"{}")),
        )
        .unwrap();
        txn.commit().unwrap();
        // One relation and its kind dictionary are durable.
        adj.record_events_logged(
            &db,
            &log,
            &[
                plain("$a"),
                plain("$b"),
                plain("$c"),
                related("$b", "m.annotation"),
            ],
            AppendDurability::Deferred,
        )
        .unwrap();
        db.coordinator().sync().unwrap();
        let durable_kinds = adj.kind_counter_for_test(&db).unwrap();
        // A later custom kind allocation and its row reach the tail before
        // their WAL group does.
        adj.record_events_logged(
            &db,
            &log,
            &[related("$c", "org.example.rolled-back-kind")],
            AppendDurability::Deferred,
        )
        .unwrap();
        assert!(
            adj.relation_of(&db, "$c", &AlwaysVisible)
                .unwrap()
                .is_some(),
            "the relation row exists before the simulated crash"
        );
        assert!(adj.kind_counter_for_test(&db).unwrap() > durable_kinds);
        adj.sync_tail(&db).unwrap();

        crash_image(&db, &root, &image);
        cut_segment_image_to_durable_mark(&image.join("wal.bin")).unwrap();
        drop(db);

        let db = Database::open(image.clone()).unwrap();
        CsrAdjacency::recover_edge_log(&db, &log).unwrap();
        assert_eq!(adj.kind_counter_for_test(&db).unwrap(), durable_kinds);
        assert_eq!(
            adj.relation_of(&db, "$b", &AlwaysVisible).unwrap(),
            Some(Relation {
                target: "$a".to_owned(),
                rel_type: "m.annotation".to_owned(),
            }),
            "a relation whose kind is durable survives the repair"
        );
        assert_eq!(
            adj.relation_of(&db, "$c", &AlwaysVisible).unwrap(),
            None,
            "the relation row must not retain a kind whose allocation rolled back"
        );
        drop(db);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(image);
    }

    #[test]
    fn rows_lost_with_the_tail_are_rebuilt_from_the_log() {
        let root = test_root("edge-log-recover");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        adj.record_events_logged(
            &db,
            &log,
            &[event("$create", &[]), event("$msg", &["$create"])],
            AppendDurability::Deferred,
        )
        .unwrap();
        adj.record_events_logged(
            &db,
            &log,
            &[event("$next", &["$msg"])],
            AppendDurability::Deferred,
        )
        .unwrap();
        assert_eq!(
            adj.prev_of(&db, "$next").unwrap(),
            Some(vec!["$msg".to_owned()])
        );

        let db = crash_keeping_only_the_wal(&root, db);
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        assert_eq!(
            adj.prev_of(&db, "$next").unwrap(),
            None,
            "the unsynced rows are gone"
        );

        let replayed = CsrAdjacency::recover_edge_log(&db, &log).unwrap();

        assert_eq!(replayed, 3, "every logged event is replayed");
        assert_eq!(
            adj.prev_of(&db, "$next").unwrap(),
            Some(vec!["$msg".to_owned()])
        );
        assert_eq!(
            adj.prev_of(&db, "$msg").unwrap(),
            Some(vec!["$create".to_owned()])
        );
        assert_eq!(
            adj.auth_of(&db, "$msg").unwrap(),
            Some(vec!["$create".to_owned()])
        );
        let create = adj.short_id(&db, "$create").unwrap().unwrap();
        let children = adj.children_of(&db, create).unwrap().unwrap();
        assert_eq!(
            adj.resolve_event_ids(&db, &children).unwrap(),
            vec![Some("$msg".to_owned())],
            "the reverse rows are rebuilt too"
        );
        assert!(adj.verify(&db).unwrap().is_consistent());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn recovery_rebuilds_every_room_in_log_order() {
        let root = test_root("edge-log-rooms");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let a = CsrAdjacency::for_collection(&ROOM_A);
        let b = CsrAdjacency::for_collection(&ROOM_B);
        a.record_events_logged(&db, &log, &[event("$a1", &[])], AppendDurability::Deferred)
            .unwrap();
        b.record_events_logged(&db, &log, &[event("$b1", &[])], AppendDurability::Deferred)
            .unwrap();
        a.record_events_logged(
            &db,
            &log,
            &[event("$a2", &["$a1"])],
            AppendDurability::Deferred,
        )
        .unwrap();

        let db = crash_keeping_only_the_wal(&root, db);
        CsrAdjacency::recover_edge_log(&db, &log).unwrap();

        let a = CsrAdjacency::for_collection(&ROOM_A);
        let b = CsrAdjacency::for_collection(&ROOM_B);
        assert_eq!(a.prev_of(&db, "$a2").unwrap(), Some(vec!["$a1".to_owned()]));
        assert_eq!(b.prev_of(&db, "$b1").unwrap(), Some(vec![]));
        assert_eq!(b.prev_of(&db, "$a2").unwrap(), None, "rooms do not leak");
        let _ = std::fs::remove_dir_all(root);
    }

    /// An entry whose rows never reached the tail, so the watermark cannot move
    /// past it.
    fn leave_a_gap(db: &Database, log: &EdgeLog) {
        let lost = LoggedEvent {
            scope: ROOM_B,
            event_id: "$gap".to_owned(),
            prev: Vec::new(),
            auth: Vec::new(),
            relation: None,
        };
        log.append(db, &[lost]).unwrap();
    }

    #[test]
    fn recovery_keeps_writes_made_after_an_applied_purge() {
        let root = test_root("edge-log-purge-applied");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        leave_a_gap(&db, &log);
        for (id, prev) in [("$a", &[][..]), ("$b", &["$a"][..])] {
            adj.record_events_logged(&db, &log, &[event(id, prev)], AppendDurability::Deferred)
                .unwrap();
        }
        adj.purge(&db).unwrap();
        // The gap keeps the watermark behind the purge marker, and this write
        // does not go through the log at all.
        adj.record_event(&db, "$post", &[], &[], None).unwrap();
        adj.sync_tail(&db).unwrap();

        let db = restart_keeping_everything(&root, db);
        CsrAdjacency::recover_edge_log(&db, &log).unwrap();

        assert_eq!(adj.prev_of(&db, "$post").unwrap(), Some(Vec::new()));
        assert_eq!(adj.prev_of(&db, "$a").unwrap(), None);
        assert_eq!(adj.prev_of(&db, "$b").unwrap(), None);
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn recovery_applies_a_logged_purge_whose_tombstone_never_landed() {
        let root = test_root("edge-log-purge-unapplied");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        adj.record_events_logged(&db, &log, &[event("$a", &[])], AppendDurability::Deferred)
            .unwrap();
        // Crash between the marker and the tombstone.
        log.append_scope_purge(&db, ROOM_A).unwrap();
        adj.sync_tail(&db).unwrap();
        assert_eq!(adj.prev_of(&db, "$a").unwrap(), Some(Vec::new()));

        let db = restart_keeping_everything(&root, db);
        CsrAdjacency::recover_edge_log(&db, &log).unwrap();

        assert_eq!(adj.prev_of(&db, "$a").unwrap(), None);
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn an_older_purge_is_not_replayed_over_the_latest_one() {
        let root = test_root("edge-log-purge-twice");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        leave_a_gap(&db, &log);
        adj.record_events_logged(&db, &log, &[event("$a", &[])], AppendDurability::Deferred)
            .unwrap();
        adj.purge(&db).unwrap();
        adj.record_events_logged(&db, &log, &[event("$b", &[])], AppendDurability::Deferred)
            .unwrap();
        adj.purge(&db).unwrap();
        adj.record_event(&db, "$post", &[], &[], None).unwrap();
        adj.sync_tail(&db).unwrap();

        let db = restart_keeping_everything(&root, db);
        CsrAdjacency::recover_edge_log(&db, &log).unwrap();

        assert_eq!(adj.prev_of(&db, "$post").unwrap(), Some(Vec::new()));
        assert_eq!(adj.prev_of(&db, "$a").unwrap(), None);
        assert_eq!(adj.prev_of(&db, "$b").unwrap(), None);
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_reused_sequence_with_another_token_is_a_different_purge() {
        let root = test_root("edge-log-purge-token");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        adj.record_events_logged(&db, &log, &[event("$a", &[])], AppendDurability::Deferred)
            .unwrap();
        let (seq, token) = log.append_scope_purge(&db, ROOM_A).unwrap();
        // The tail names this sequence number as applied, but for a purge whose
        // log entry rolled back and whose number was then reused.
        adj.mark_purge_applied_for_test(&db, seq, token.wrapping_add(1));

        let db = restart_keeping_everything(&root, db);
        CsrAdjacency::recover_edge_log(&db, &log).unwrap();

        assert_eq!(
            adj.prev_of(&db, "$a").unwrap(),
            None,
            "the unapplied purge must not be skipped on a matching number alone"
        );
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn an_unchanged_logged_event_does_not_pin_the_watermark() {
        let root = test_root("edge-log-unchanged-watermark");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        for (id, prev) in [("$a", &[][..]), ("$a", &[][..]), ("$b", &["$a"][..])] {
            adj.record_events_logged(&db, &log, &[event(id, prev)], AppendDurability::Deferred)
                .unwrap();
        }
        assert_eq!(log.head(&db).unwrap(), 3);

        CsrAdjacency::checkpoint_edge_log(&db, &log).unwrap();

        assert_eq!(
            log.pruned(&db).unwrap(),
            3,
            "the entry for the unchanged event must not block pruning"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn recovery_replays_nothing_when_the_rows_survived() {
        let root = test_root("edge-log-idempotent");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        adj.record_events_logged(
            &db,
            &log,
            &[event("$create", &[])],
            AppendDurability::Durable,
        )
        .unwrap();

        assert_eq!(CsrAdjacency::recover_edge_log(&db, &log).unwrap(), 0);
        assert_eq!(adj.prev_of(&db, "$create").unwrap(), Some(vec![]));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_checkpoint_prunes_the_log_and_leaves_nothing_to_replay() {
        let root = test_root("edge-log-checkpoint");
        let db = Database::open(root.clone()).unwrap();
        let log = EdgeLog::new();
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        adj.record_events_logged(
            &db,
            &log,
            &[event("$create", &[]), event("$msg", &["$create"])],
            AppendDurability::Deferred,
        )
        .unwrap();
        assert_eq!(log.pruned(&db).unwrap(), 0);

        CsrAdjacency::checkpoint_edge_log(&db, &log).unwrap();

        assert_eq!(log.pruned(&db).unwrap(), log.head(&db).unwrap());
        assert_eq!(
            log.read_range(&db, 0, log.head(&db).unwrap()).unwrap(),
            Vec::new()
        );
        // The rows are durable now, so a restart replays nothing.
        let db = restart_keeping_everything(&root, db);
        assert_eq!(CsrAdjacency::recover_edge_log(&db, &log).unwrap(), 0);
        let adj = CsrAdjacency::for_collection(&ROOM_A);
        assert_eq!(
            adj.prev_of(&db, "$msg").unwrap(),
            Some(vec!["$create".to_owned()])
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn logging_needs_an_adjacency_addressed_by_its_room_collection() {
        let root = test_root("edge-log-needs-collection");
        let db = Database::open(root.clone()).unwrap();
        let by_room_id = CsrAdjacency::new(ROOM);

        let result = by_room_id.record_events_logged(
            &db,
            &EdgeLog::new(),
            &[event("$create", &[])],
            AppendDurability::Deferred,
        );

        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(root);
    }
}
