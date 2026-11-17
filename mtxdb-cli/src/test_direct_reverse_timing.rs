//! Throwaway timing of the direct-import reverse transpose.
//!
//! Splits `ensure_reverse` (the whole-room `PREV` + `AUTH` transpose a late
//! arrival forces) into its phases, then times `dirty_closure` on the cached
//! maps. Test-only and ignored; it changes no production code.
//!
//! Run: `cargo +1.90.0 test --release -p mtxdb-cli direct_reverse_timing -- --ignored --nocapture`

use super::*;
use std::time::{Duration, Instant};

const ROOM: &str = "!direct-timing:example.org";
const MEMBERS: usize = 100;
const BATCH: usize = 256;

#[allow(clippy::arithmetic_side_effects)]
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

/// `reverse_edges_by_event_id`, replicated phase by phase.
fn phases(
    db: &Database,
    adjacency: &CsrAdjacency,
    family: mtxdb::csr::EdgeFamily,
) -> (Duration, Duration, Duration, usize) {
    let t = Instant::now();
    let reverse = adjacency.reverse_family(db, family).unwrap();
    let derive = t.elapsed();

    let t = Instant::now();
    let parents = reverse.parents().to_vec();
    let mut short_ids = parents.clone();
    for parent in &parents {
        if let Some(children) = reverse.children_of(*parent) {
            short_ids.extend(children.iter().map(|edge| edge.target));
        }
    }
    short_ids.sort_unstable();
    short_ids.dedup();
    let resolved = adjacency.resolve_event_ids(db, &short_ids).unwrap();
    let resolve = t.elapsed();

    let t = Instant::now();
    let names: HashMap<u32, String> = short_ids
        .iter()
        .copied()
        .zip(resolved)
        .filter_map(|(short, name)| name.map(|name| (short, name)))
        .collect();
    let mut edges: HashMap<String, Vec<String>> = HashMap::with_capacity(parents.len());
    for parent in &parents {
        let Some(parent_name) = names.get(parent) else {
            continue;
        };
        let mut children: Vec<String> = reverse
            .children_of(*parent)
            .unwrap_or(&[])
            .iter()
            .filter_map(|edge| names.get(&edge.target).cloned())
            .collect();
        children.sort();
        children.dedup();
        edges.insert(parent_name.clone(), children);
    }
    let build = t.elapsed();
    (derive, resolve, build, edges.len())
}

#[test]
#[ignore = "timing measurement, not a correctness test"]
#[allow(clippy::too_many_lines, clippy::arithmetic_side_effects)]
fn direct_reverse_timing() {
    for events in [10_000_usize, 100_000] {
        let root = std::env::temp_dir().join(format!(
            "mtxdb-direct-reverse-timing-{}-{events}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let db = Database::open(root.clone()).unwrap();
        let adj = CsrAdjacency::new(ROOM);
        let name = |n: usize| format!("${:016x}", (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        // 0 = create, 1 = power levels, 2..2+MEMBERS = members, then messages in
        // a prev chain; every message cites create, power levels and one member.
        let first = 2 + MEMBERS;
        let total = first + events;
        let names: Vec<String> = (0..total).map(name).collect();
        let mut written = 0;
        while written < total {
            let end = (written + BATCH).min(total);
            let prevs: Vec<Vec<&str>> = (written..end)
                .map(|n| {
                    if n == 0 {
                        vec![]
                    } else {
                        vec![names[n - 1].as_str()]
                    }
                })
                .collect();
            let auths: Vec<Vec<&str>> = (written..end)
                .map(|n| {
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
                })
                .collect();
            let records: Vec<EventRecord<'_>> = (written..end)
                .map(|n| EventRecord {
                    event_id: names[n].as_str(),
                    prev: &prevs[n - written],
                    auth: &auths[n - written],
                    relation: None,
                })
                .collect();
            adj.record_events(&db, &records).unwrap();
            written = end;
        }
        println!(
            "direct import transpose, {events} events, {} sealed segments:",
            count_segments(&root)
        );

        for (label, family) in [("PREV", PREV), ("AUTH", AUTH)] {
            let (derive, resolve, build, parents) = phases(&db, &adj, family);
            println!(
                "  {label}: derive {derive:?} | resolve ids {resolve:?} | build maps {build:?} | {parents} parents"
            );
        }

        let collection_id = [7_u8; 16];
        let mut cache = ImportRunCache::default();
        let t = Instant::now();
        let reverse = cache.ensure_reverse(&db, &collection_id, ROOM).unwrap();
        let cold = t.elapsed();
        let (prev_parents, auth_parents) = (reverse.prev.len(), reverse.auth.len());
        let t = Instant::now();
        let _ = cache.ensure_reverse(&db, &collection_id, ROOM).unwrap();
        let warm = t.elapsed();
        println!(
            "  ensure_reverse cold {cold:?} ({prev_parents} prev / {auth_parents} auth parents) | cached {warm:?}"
        );

        let hit = |ids: &[&str], prev: bool| {
            let set: HashSet<String> = ids.iter().map(|s| (*s).to_owned()).collect();
            if prev {
                WaitingHits {
                    prev: set,
                    auth: HashSet::new(),
                }
            } else {
                WaitingHits {
                    prev: HashSet::new(),
                    auth: set,
                }
            }
        };
        let cases = [
            (
                "prev hit 10 from the tip",
                hit(&[names[total - 11].as_str()], true),
            ),
            (
                "auth hit on a member event",
                hit(&[names[2].as_str()], false),
            ),
            ("auth hit on create", hit(&[names[0].as_str()], false)),
        ];
        for (label, hits) in &cases {
            let t = Instant::now();
            let dirty = cache
                .dirty_closure(&db, &collection_id, ROOM, hits)
                .unwrap();
            let elapsed = t.elapsed();
            let over_half = dirty.len().saturating_mul(2) > total;
            println!(
                "  dirty_closure ({label}): {} events in {elapsed:?}, exceeds half the room: {over_half}",
                dirty.len()
            );
        }
        let _ = std::fs::remove_dir_all(root);
    }
}
