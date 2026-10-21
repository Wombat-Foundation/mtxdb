#![allow(clippy::tests_outside_test_module)]

//! Pure `AuthGraph` closure semantics: ancestors-only, memoized over diamonds,
//! union/`include_given`, and explicit incomplete/cycle reporting. These need
//! only the `bitmaps` feature and no storage engine.

use mtxdb::auth_closure::{AuthGraph, ClosureOutcome};
use mtxdb::{BitmapSet, DomainTag};

fn domain() -> DomainTag {
    DomainTag::derive(b"auth-graph-tests")
}

/// A diamond: `create` is an auth parent of both `member` and `power`, and
/// `msg` auths all three. Short ids are fixed here, not allocated.
fn diamond() -> AuthGraph {
    let mut graph = AuthGraph::new(domain());
    graph.insert_event("$create", 1);
    graph.record_auth(1, []);
    graph.insert_event("$member", 2);
    graph.record_auth(2, [1]);
    graph.insert_event("$power", 3);
    graph.record_auth(3, [1]);
    graph.insert_event("$msg", 4);
    graph.record_auth(4, [1, 2, 3]);
    graph
}

fn set(values: &[u32]) -> BitmapSet {
    BitmapSet::from_values(domain(), values.iter().copied())
}

#[test]
fn closure_is_ancestors_only_and_unions_the_diamond() {
    let graph = diamond();
    assert_eq!(
        graph.compute("$msg").unwrap(),
        ClosureOutcome::Complete(set(&[1, 2, 3])),
        "closure is every ancestor, and not the queried event itself"
    );
    assert_eq!(
        graph.compute("$create").unwrap(),
        ClosureOutcome::Complete(set(&[])),
        "a recorded leaf has a complete, empty closure"
    );
}

#[test]
fn graph_is_memoized_across_a_diamond_without_a_false_cycle() {
    // Computing the shared parent first, then the child that also reaches it,
    // must not be mistaken for a cycle.
    let graph = diamond();
    assert_eq!(
        graph.compute("$member").unwrap(),
        ClosureOutcome::Complete(set(&[1]))
    );
    assert_eq!(
        graph.compute("$msg").unwrap(),
        ClosureOutcome::Complete(set(&[1, 2, 3]))
    );
}

#[test]
fn union_of_excludes_given_events_and_with_given_includes_them() {
    let graph = diamond();
    assert_eq!(
        graph.union_of(&["$msg", "$member"]).unwrap(),
        ClosureOutcome::Complete(set(&[1, 2, 3]))
    );
    assert_eq!(
        graph.union_of_with_given(&["$msg"]).unwrap(),
        ClosureOutcome::Complete(set(&[1, 2, 3, 4])),
        "include_given adds each queried event's own id"
    );
}

#[test]
fn missing_parent_is_reported_by_event_id() {
    let mut graph = AuthGraph::new(domain());
    graph.insert_event("$gone", 7);
    // `$gone` is referenced but has no recorded auth adjacency.
    graph.insert_event("$orphan", 6);
    graph.record_auth(6, [7]);
    assert_eq!(
        graph.compute("$orphan").unwrap(),
        ClosureOutcome::Incomplete {
            missing: vec!["$gone".to_owned()]
        }
    );
}

#[test]
fn unknown_event_is_incomplete_and_named() {
    let graph = diamond();
    assert_eq!(
        graph.compute("$never-seen").unwrap(),
        ClosureOutcome::Incomplete {
            missing: vec!["$never-seen".to_owned()]
        }
    );
}

#[test]
fn auth_cycle_is_corruption() {
    let mut graph = AuthGraph::new(domain());
    graph.insert_event("$a", 1);
    graph.record_auth(1, [2]);
    graph.insert_event("$b", 2);
    graph.record_auth(2, [1]);
    assert!(
        graph.compute("$a").is_err(),
        "an auth cycle is corruption, not an incomplete closure"
    );
}

#[test]
fn closures_are_tagged_with_the_graph_domain() {
    let ClosureOutcome::Complete(closure) = diamond().compute("$msg").unwrap() else {
        panic!("complete closure");
    };
    assert_eq!(closure.domain(), domain());
    assert!(closure.contains(1));
    assert!(!closure.contains(4), "the queried event is excluded");
}
