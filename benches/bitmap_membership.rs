//! Encoded closure membership: full decode versus the validating probe.
//!
//! `BitmapSet::decode_in_domain` materializes every container before one
//! membership test; `BitmapSet::contains_encoded` validates the payload and
//! probes it without allocating. Both read the whole closure blob, so this
//! measures decode/allocation work, not bytes read.
//!
//! Inputs are black-boxed so the compiler cannot hoist the blob or the query
//! value out of the measurement. Dense, sparse and mixed sets exercise array
//! and bitset containers at different cardinalities.
//!
//! Usage: `bitmap_membership`

use std::hint::black_box;
use std::time::Instant;

use mtxdb::{BitmapSet, DomainTag};

const DOMAIN: DomainTag = DomainTag::new([7u8; 16]);

struct Case {
    kind: &'static str,
    values: Vec<u32>,
}

/// Dense, sparse and mixed memberships over the same ordinal span.
fn cases(span: u32) -> Vec<Case> {
    let dense: Vec<u32> = (0..span).collect();
    let sparse: Vec<u32> = (0..span).step_by(64).collect();
    let mut mixed: Vec<u32> = (0..span / 2)
        .chain((span / 2..span).step_by(64))
        .collect();
    mixed.dedup();

    vec![
        Case { kind: "dense", values: dense },
        Case { kind: "sparse", values: sparse },
        Case { kind: "mixed", values: mixed },
    ]
}

/// Hit and miss values spread across the set.
fn queries(values: u32) -> (Vec<u32>, Vec<u32>) {
    let step = (values / 512).max(1);
    let hits = (0..values).step_by(step as usize).collect();
    let misses = (values..values.saturating_mul(2))
        .step_by(step as usize)
        .collect();
    (hits, misses)
}

fn time_queries(values: &[u32], iterations: u32, mut query: impl FnMut(u32) -> bool) -> f64 {
    // Warm up before timing.
    for &value in values {
        black_box(query(black_box(value)));
    }
    let start = Instant::now();
    for _ in 0..iterations {
        for &value in values {
            black_box(query(black_box(value)));
        }
    }
    let per_query = start.elapsed().as_nanos() as f64 / f64::from(iterations);
    per_query / values.len() as f64 / 1e3
}

fn main() {
    println!(
        "{:<10} {:<8} {:>10} {:>12} {:>15} {:>15} {:>16} {:>16}",
        "span",
        "kind",
        "values",
        "blob bytes",
        "decode hit us",
        "probe hit us",
        "decode miss us",
        "probe miss us"
    );

    for span in [100_000u32, 1_000_000] {
        for case in cases(span) {
            let set = BitmapSet::from_values(DOMAIN, case.values.iter().copied());
            let bytes = set.encode().expect("encode");
            let (hits, misses) = queries(span);
            let iterations = if case.values.len() > 500_000 { 3 } else { 10 };

            let decode_hit = time_queries(&hits, iterations, |value| {
                BitmapSet::decode_in_domain(black_box(&bytes), DOMAIN)
                    .expect("decode")
                    .contains(black_box(value))
            });
            let probe_hit = time_queries(&hits, iterations, |value| {
                BitmapSet::contains_encoded(black_box(&bytes), DOMAIN, black_box(value))
                    .expect("probe")
            });
            let decode_miss = time_queries(&misses, iterations, |value| {
                BitmapSet::decode_in_domain(black_box(&bytes), DOMAIN)
                    .expect("decode")
                    .contains(black_box(value))
            });
            let probe_miss = time_queries(&misses, iterations, |value| {
                BitmapSet::contains_encoded(black_box(&bytes), DOMAIN, black_box(value))
                    .expect("probe")
            });

            println!(
                "{span:<10} {:<8} {:>10} {:>12} {decode_hit:>15.3} {probe_hit:>15.3} {decode_miss:>16.3} {probe_miss:>16.3}",
                case.kind,
                case.values.len(),
                bytes.len()
            );
        }
    }
}
