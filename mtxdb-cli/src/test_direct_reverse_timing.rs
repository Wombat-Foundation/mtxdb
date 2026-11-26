//! Throwaway timing of the direct-import reverse transpose.
//!
//! Splits `ensure_reverse` (the whole-room `PREV` + `AUTH` transpose a late
//! arrival forces) into its phases, then times `dirty_closure` on the cached
//! maps. Runs two fixtures: a `PREV` chain and an AUTH-only star with no
//! `prev_events`, because the two give different dirty-closure shapes.
//!
//! Test-only and ignored; it changes no production code.
//!
//! Run: `cargo +1.90.0 test --release -p mtxdb-cli direct_reverse_timing -- --ignored --nocapture`

use super::*;
use rezzy::json::Value as OwnedValue;
use std::time::{Duration, Instant};

const ROOM: &str = "!direct-timing:example.org";
const MEMBERS: usize = 100;
const BATCH: usize = 256;

#[allow(clippy::arithmetic_side_effects)]
fn count_segments(dir: &std::path::Path) -> usize {
    let mut count = 0_usize;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|extension| extension == "csr") {
                count = count.saturating_add(1);
            }
        }
    }
    count
}

fn render_ids(ids: &[&str]) -> String {
    ids.iter()
        .map(|event_id| format!("\"{event_id}\""))
        .collect::<Vec<_>>()
        .join(",")
}

fn event_value(id: &str, prev: &[&str], auth: &[&str]) -> OwnedValue {
    let json = format!(
        r#"{{"event_id":"{id}","room_id":"{ROOM}","prev_events":[{}],"auth_events":[{}]}}"#,
        render_ids(prev),
        render_ids(auth)
    );
    OwnedValue::parse_bytes(json.as_bytes()).expect("benchmark event is valid JSON")
}

/// Replicate `reverse_edges_by_event_id` with a timer around each phase.
fn phases(
    db: &Database,
    adjacency: &CsrAdjacency,
    family: mtxdb::csr::EdgeFamily,
) -> (Duration, Duration, Duration, Duration, usize) {
    let started = Instant::now();
    let reverse = adjacency.reverse_family(db, family).unwrap();
    let derive = started.elapsed();

    let started = Instant::now();
    let parents = reverse.parents().to_vec();
    let mut short_ids = parents.clone();
    for parent in &parents {
        if let Some(children) = reverse.children_of(*parent) {
            short_ids.extend(children.iter().map(|edge| edge.target));
        }
    }
    short_ids.sort_unstable();
    short_ids.dedup();
    let collect_ids = started.elapsed();

    let started = Instant::now();
    let resolved = adjacency.resolve_event_ids(db, &short_ids).unwrap();
    let resolve_ids = started.elapsed();

    let started = Instant::now();
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
    let build_maps = started.elapsed();
    (derive, collect_ids, resolve_ids, build_maps, edges.len())
}

#[test]
#[ignore = "timing measurement, not a correctness test"]
#[allow(clippy::too_many_lines, clippy::arithmetic_side_effects)]
fn direct_reverse_timing() {
    for (fixture, prev_chain) in [("prev chain", true), ("no prev", false)] {
        for messages in [10_000_usize, 100_000] {
            let root = crate::test_scratch::root().join(format!(
                "mtxdb-direct-reverse-timing-{}-{}-{messages}",
                std::process::id(),
                if prev_chain { "chain" } else { "star" }
            ));
            let _ = std::fs::remove_dir_all(&root);
            let db = Database::open(root.clone()).unwrap();
            let adjacency = CsrAdjacency::new(ROOM);
            let template = default_matrix_import_template();
            let name = |number: usize| format!("${:016x}", number as u64);

            // Match auth_star_timing: create, power levels, 100 members, then
            // messages citing create, power levels and one member. `prev` is a
            // chain for the first fixture and empty for the second.
            let first_message = 2 + MEMBERS;
            let total = first_message + messages;
            let names: Vec<String> = (0..total).map(name).collect();
            let mut written = 0_usize;
            while written < total {
                let end = (written + BATCH).min(total);
                let events: Vec<OwnedValue> = (written..end)
                    .map(|number| {
                        let auth: Vec<&str> = if number == 0 {
                            Vec::new()
                        } else if number < first_message {
                            vec![names[0].as_str()]
                        } else {
                            vec![
                                names[0].as_str(),
                                names[1].as_str(),
                                names[2 + number % MEMBERS].as_str(),
                            ]
                        };
                        let prev: Vec<&str> = if prev_chain && number > 0 {
                            vec![names[number - 1].as_str()]
                        } else {
                            Vec::new()
                        };
                        event_value(&names[number], &prev, &auth)
                    })
                    .collect();
                record_csr_adjacency(&db, &template, &events).unwrap();
                written = end;
            }

            println!(
                "direct import reverse, {fixture}, {messages} messages / {total} total events, {} sealed segments (one run):",
                count_segments(&root)
            );
            let collection_id = [7_u8; 16];
            let mut cache = ImportRunCache::default();
            let started = Instant::now();
            let reverse = cache.ensure_reverse(&db, &collection_id, ROOM).unwrap();
            let cache_miss = started.elapsed();
            let (prev_parents, auth_parents) = (reverse.prev.len(), reverse.auth.len());
            let started = Instant::now();
            cache.ensure_reverse(&db, &collection_id, ROOM).unwrap();
            let warm = started.elapsed();
            println!(
                "  ensure_reverse: cache miss {cache_miss:?} ({prev_parents} PREV / {auth_parents} AUTH parents), cached {warm:?}"
            );

            for (label, family) in [("PREV", PREV), ("AUTH", AUTH)] {
                let (derive, collect, resolve, build, parent_count) =
                    phases(&db, &adjacency, family);
                println!(
                    "  {label}: derive {derive:?} | collect ids {collect:?} | resolve ids {resolve:?} | build string maps {build:?} | {parent_count} parents"
                );
            }

            let hits = |prev: &[&str], auth: &[&str]| WaitingHits {
                prev: prev.iter().map(|id| (*id).to_owned()).collect(),
                auth: auth.iter().map(|id| (*id).to_owned()).collect(),
            };
            let cases = [
                (
                    "prev hit 10 from the tip",
                    hits(&[names[total - 11].as_str()], &[]),
                ),
                ("auth hit on a member", hits(&[], &[names[2].as_str()])),
                ("auth hit on create", hits(&[], &[names[0].as_str()])),
            ];
            for (label, hits) in &cases {
                let started = Instant::now();
                let dirty = cache
                    .dirty_closure(&db, &collection_id, ROOM, hits)
                    .unwrap();
                let elapsed = started.elapsed();
                let over_half = dirty.len().saturating_mul(2) > total;
                println!(
                    "  cached dirty_closure ({label}): {} events in {elapsed:?}; exceeds half-room fallback: {over_half}",
                    dirty.len()
                );
            }
            drop(db);
            let _ = std::fs::remove_dir_all(root);
        }
    }
}
