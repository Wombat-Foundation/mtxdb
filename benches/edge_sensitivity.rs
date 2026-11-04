//! Controlled codec sensitivity cases; these are not workload measurements.
use super::{append_reverse_row, BitmapSet, CsrRawForward, CsrVarintForward, BENCH_DOMAIN};

pub(super) fn run() {
    density();
    holes();
    forward();
    roots();
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
