//! Controlled codec sensitivity cases; these are not workload measurements.
use super::{append_reverse_row, BitmapSet, CsrRawForward, CsrVarintForward, BENCH_DOMAIN};

pub(super) fn run() {
    density();
    pl_hubs();
    holes();
    forward();
    roots();
    state_anchors();
}

/// State-group annotation density and run fragmentation on a synthetic DAG.
///
/// This is a size model, not a workload measurement. Events are assigned to
/// `branches` concurrent tips round-robin; every `merge_period` events a merge
/// joins all live tips. A state event starts a new group, a non-state event
/// inherits its single parent's group, and a merge whose parent groups differ
/// starts a new group (content deduplication is deliberately not modelled).
/// Anchors are the DAG facts; runs are the exact RLE of the resulting
/// per-event sequence in short-id order.
fn state_anchors() {
    println!(
        "# state_anchors: 8192-event synthetic DAG; 32-byte group tokens; 8 B/run model; no content dedupe"
    );
    println!(
        "branches\tstate_pct\tevents\tstate_events\tmerge_events\tmerge_anchors\tanchors\truns\truns_per_1024\tdistinct_groups\tflat_B\tpacked_B\trle_B\trle_B_per_event"
    );
    for branches in [1usize, 2, 4, 8, 16] {
        for state_percent in [0u32, 1, 5, 20, 100] {
            state_anchor_row(branches, state_percent);
        }
    }
}

fn state_anchor_row(branches: usize, state_percent: u32) {
    const EVENTS: usize = 8192;
    const MERGE_PERIOD: usize = 256;
    let mut rng = super::PseudoRng(
        0x5EED ^ u64::try_from(branches).expect("branches fit") ^ (u64::from(state_percent) << 32),
    );
    let mut groups: Vec<u32> = Vec::with_capacity(EVENTS);
    let mut tips: Vec<Option<u32>> = vec![None; branches];
    let mut next_group = 0u32;
    let mut state_events = 0usize;
    let mut merge_events = 0usize;
    let mut merge_anchors = 0usize;
    let mut root_anchors = 0usize;

    for event in 0..EVENTS {
        if event > 0 && event % MERGE_PERIOD == 0 {
            merge_events = merge_events.saturating_add(1);
            let mut parent_groups = tips.iter().flatten().copied();
            let first = parent_groups.next();
            let all_equal = first.is_none_or(|group| parent_groups.all(|other| other == group));
            let group = if all_equal {
                first.unwrap_or(0)
            } else {
                merge_anchors = merge_anchors.saturating_add(1);
                let group = next_group;
                next_group = next_group.saturating_add(1);
                group
            };
            groups.push(group);
            tips.fill(Some(group));
            continue;
        }

        let branch = event
            .checked_rem(branches)
            .expect("branch count is nonzero");
        let is_state = rng.below(100) < u64::from(state_percent);
        let group = match (is_state, tips[branch]) {
            (true, _) => {
                state_events = state_events.saturating_add(1);
                let group = next_group;
                next_group = next_group.saturating_add(1);
                group
            }
            (false, Some(parent)) => parent,
            (false, None) => {
                root_anchors = root_anchors.saturating_add(1);
                let group = next_group;
                next_group = next_group.saturating_add(1);
                group
            }
        };
        groups.push(group);
        tips[branch] = Some(group);
    }

    let runs = groups
        .windows(2)
        .filter(|pair| pair[0] != pair[1])
        .count()
        .saturating_add(usize::from(!groups.is_empty()));
    // Parentless first events on a branch open a group too, so they anchor.
    let anchors = state_events
        .saturating_add(merge_anchors)
        .saturating_add(root_anchors);
    let distinct = usize::try_from(next_group).expect("group count fits");
    let bits = usize::BITS
        .checked_sub(distinct.saturating_sub(1).leading_zeros())
        .expect("bit width fits");
    let dictionary = distinct.checked_mul(32).expect("size fits");
    let flat_bytes = EVENTS.checked_mul(32).expect("size fits");
    let packed_bytes = dictionary
        .checked_add(
            EVENTS
                .checked_mul(usize::try_from(bits).expect("bits fit"))
                .expect("size fits")
                .div_ceil(8),
        )
        .expect("size fits");
    let rle_bytes = dictionary
        .checked_add(runs.checked_mul(8).expect("runs fit"))
        .expect("size fits");
    let runs_per_1024 = runs.saturating_mul(1024) / EVENTS;
    println!(
        "{branches}\t{state_percent}\t{EVENTS}\t{state_events}\t{merge_events}\t{merge_anchors}\t{anchors}\t{runs}\t{runs_per_1024}\t{distinct}\t{flat_bytes}\t{packed_bytes}\t{rle_bytes}\t{:.2}",
        super::len_f64(rle_bytes) / super::len_f64(EVENTS)
    );
}

/// Power-level-like rows: a few contiguous authorization intervals per hub,
/// broken by evenly sized fork/backfill gaps. The run size stays a 4 B-per-run
/// model; gap and bitmap sizes are measured by the real encoders.
fn pl_hubs() {
    println!("# pl_hubs: piecewise-contiguous rows with 2-short fork gaps; run_B = 4*runs model");
    println!("children\truns\tgap_B\tbitmap_B\trun_B\twinner\tmax_runs_vs_gap\tmax_runs_vs_bitmap");
    for children in [4096usize, 8192, 16_384, 32_768, 60_000] {
        for runs in [1usize, 4, 16, 64, 256, 1024, 2048, 4096] {
            let Some(values) = piecewise_children(children, runs) else {
                continue;
            };
            let mut gaps = Vec::new();
            append_reverse_row(0, &values, &mut gaps);
            let bitmap = BitmapSet::from_values(BENCH_DOMAIN, values.iter().copied())
                .encode()
                .expect("bitmap encodes");
            let run_bytes = runs.checked_mul(4).expect("size fits");
            let (_, winner) = [
                (gaps.len(), "gap"),
                (bitmap.len(), "bitmap"),
                (run_bytes, "run"),
            ]
            .into_iter()
            .min()
            .expect("three candidates");
            // Largest run count whose 4 B model still beats each measured encoding.
            let max_vs_gap = gaps
                .len()
                .saturating_sub(1)
                .checked_div(4)
                .expect("nonzero divisor");
            let max_vs_bitmap = bitmap
                .len()
                .saturating_sub(1)
                .checked_div(4)
                .expect("nonzero divisor");
            println!(
                "{children}\t{runs}\t{}\t{}\t{run_bytes}\t{winner}\t{max_vs_gap}\t{max_vs_bitmap}",
                gaps.len(),
                bitmap.len()
            );
        }
    }
}

/// `children` shorts split into `runs` near-equal ranges, separated by two
/// missing shorts. Returns `None` when there are more runs than children.
fn piecewise_children(children: usize, runs: usize) -> Option<Vec<u32>> {
    let base = children.checked_div(runs)?;
    let extra = children.checked_rem(runs)?;
    if base == 0 {
        return None;
    }
    let mut values = Vec::with_capacity(children);
    let mut next = 0u32;
    for index in 0..runs {
        let length = base.checked_add(usize::from(index < extra))?;
        for _ in 0..length {
            values.push(next);
            next = next.checked_add(1)?;
        }
        next = next.checked_add(2)?;
    }
    Some(values)
}

fn density() {
    println!(
        "# density: serialized bitmap versus reverse gap payload; run bytes are a size model only"
    );
    println!("pattern\tchildren\tgap_B\tbitmap_B\trun_model_B\tsmaller_measured_codec");
    for stride in [1u32, 2, 4, 8, 16, 32, 64, 128, 256] {
        let children: Vec<u32> = (0..65_536)
            .step_by(usize::try_from(stride).expect("stride fits"))
            .collect();
        density_row(&format!("stride={stride}"), &children);
    }
    for length in [16u32, 64, 256, 1024, 4096, 8192, 8229, 8230, 16_384, 65_536] {
        let children: Vec<u32> = (0..length).collect();
        density_row(&format!("consecutive={length}"), &children);
    }
}

fn density_row(pattern: &str, children: &[u32]) {
    let mut gaps = Vec::new();
    append_reverse_row(0, children, &mut gaps);
    let bitmap = BitmapSet::from_values(BENCH_DOMAIN, children.iter().copied())
        .encode()
        .expect("bitmap encodes");
    let runs = children
        .windows(2)
        .filter(|pair| pair[0].checked_add(1) != Some(pair[1]))
        .count()
        .checked_add(usize::from(!children.is_empty()))
        .expect("run count fits");
    let run_bytes = runs.checked_mul(4).expect("size fits");
    let winner = match gaps.len().cmp(&bitmap.len()) {
        std::cmp::Ordering::Less => "gap",
        std::cmp::Ordering::Equal => "tie",
        std::cmp::Ordering::Greater => "bitmap",
    };
    println!(
        "{pattern}\t{}\t{}\t{}\t{run_bytes}\t{winner}",
        children.len(),
        gaps.len(),
        bitmap.len()
    );
}

fn holes() {
    println!("# holes: one 65536-short universe; raw directory payloads versus serialized bitmap (includes headers); values excluded");
    println!("hole_percent\tpresent\tsorted_u16_B\tgap_B\tbitmap_B\tsmaller_directory");
    for hole_percent in [0u32, 25, 50, 75, 85, 87, 88, 90, 93, 94, 95, 99, 100] {
        let present: Vec<u32> = (0u32..65_536)
            .filter(|short| short.checked_rem(100).expect("nonzero divisor") >= hole_percent)
            .collect();
        let sparse_bytes = present.len().checked_mul(2).expect("size fits");
        let mut gaps = Vec::new();
        append_reverse_row(0, &present, &mut gaps);
        let bitmap_bytes = BitmapSet::from_values(BENCH_DOMAIN, present.iter().copied())
            .encode()
            .expect("bitmap encodes")
            .len();
        let (_, winner) = [
            (sparse_bytes, "u16"),
            (gaps.len(), "gaps"),
            (bitmap_bytes, "bitmap"),
        ]
        .into_iter()
        .min()
        .expect("three candidates");
        println!(
            "{hole_percent}\t{}\t{sparse_bytes}\t{}\t{bitmap_bytes}\t{winner}",
            present.len(),
            gaps.len()
        );
    }
}

fn forward() {
    println!("# forward: 8192 rows, one edge each; requested late-parent percent and distance");
    println!("late_percent\tdistance\tnegative_deltas\traw_CSR_B\tvarint_payload_B");
    for late_percent in [0u32, 1, 10, 50, 100] {
        for distance in [1u32, 8, 128, 1024, 4096] {
            let edges: Vec<Vec<u32>> = (0u32..8192)
                .map(|child| {
                    let parent = if child.checked_rem(100).expect("nonzero divisor") < late_percent
                    {
                        child
                            .checked_add(distance)
                            .expect("small shorts fit")
                            .min(8191)
                    } else {
                        child.saturating_sub(distance)
                    };
                    vec![parent]
                })
                .collect();
            let negative = edges
                .iter()
                .enumerate()
                .filter(|(child, parents)| {
                    usize::try_from(parents[0]).expect("short fits") > *child
                })
                .count();
            let varint = CsrVarintForward::build(&edges);
            let mut decoded = Vec::new();
            for (short, expected) in edges.iter().enumerate() {
                varint.lookup(u32::try_from(short).expect("short fits"), &mut decoded);
                assert_eq!(
                    decoded, *expected,
                    "sensitivity row {short}, late={late_percent}, distance={distance}"
                );
            }
            let raw = CsrRawForward::build(&edges);
            println!(
                "{late_percent}\t{distance}\t{negative}\t{}\t{}",
                raw.total_bytes(),
                varint.total_bytes()
            );
        }
    }
}

fn roots() {
    println!(
        "# roots: byte-size model, 1024 rows and synthetic 32-byte root tokens; metadata excluded"
    );
    println!("distinct\tpattern\truns\tflat_B\tdictionary_packed_B\trle_dictionary_B");
    for distinct in [1usize, 4, 16, 64, 256, 1024] {
        let bits = usize::try_from(
            usize::BITS
                .checked_sub(distinct.saturating_sub(1).leading_zeros())
                .expect("bit width fits"),
        )
        .expect("bits fit");
        let dictionary = distinct.checked_mul(32).expect("size fits");
        let packed = dictionary
            .checked_add(1024usize.checked_mul(bits).expect("bits fit").div_ceil(8))
            .expect("size fits");
        for (pattern, runs) in [
            ("clustered", distinct),
            ("alternating", if distinct == 1 { 1 } else { 1024 }),
        ] {
            let rle = dictionary
                .checked_add(runs.checked_mul(8).expect("runs fit"))
                .expect("size fits");
            println!("{distinct}\t{pattern}\t{runs}\t32768\t{packed}\t{rle}");
        }
    }
}
