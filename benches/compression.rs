//! Raw-versus-zstd write-cost microbenchmark for packfile frames.
//!
//! Run with `cargo bench --bench compression` from `benches/`. This measures
//! CPU-side frame production only: output goes to a black-box discard writer so filesystem
//! latency, shard rotation, index insertion, and fsync do not obscure the
//! per-record cost of `write_record`'s unconditional zstd attempt.
//!
//! The generated ``hamt-shaped'' payloads contain a small fixed header and
//! 16-byte content-address-like references. They are not a substitute for a
//! captured Synapse HAMT-node corpus; use this benchmark to quantify the codec
//! overhead by size and entropy, then validate its payload mix against an
//! integration trace.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::pedantic,
    clippy::uninlined_format_args
)]

use std::io::{self, Write};
use std::time::{Duration, Instant};

use bytes::Bytes;
use mtxdb::packfile::{self, Record};

const RECORDS_PER_CASE: usize = 25_000;
const COLLECTION_ID: [u8; 16] = [0xC0; 16];

#[derive(Clone, Copy)]
enum PayloadKind {
    HamtShaped,
    MatrixJsonLike,
    Incompressible,
}

impl PayloadKind {
    const ALL: [Self; 3] = [Self::HamtShaped, Self::MatrixJsonLike, Self::Incompressible];

    const fn label(self) -> &'static str {
        match self {
            Self::HamtShaped => "HAMT-shaped",
            Self::MatrixJsonLike => "Matrix-JSON-like",
            Self::Incompressible => "incompressible",
        }
    }
}

/// A small deterministic mixer. It makes reference bytes look like real
/// content hashes while keeping every benchmark run reproducible.
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn record_id(index: usize) -> [u8; 16] {
    let mut id = [0_u8; 16];
    let first = splitmix64(index as u64);
    id[..8].copy_from_slice(&first.to_le_bytes());
    id[8..].copy_from_slice(&splitmix64(first).to_le_bytes());
    id
}

fn payload(kind: PayloadKind, len: usize, seed: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(len);
    match kind {
        PayloadKind::HamtShaped => {
            // A compact node header followed by hash-sized references. The
            // headers repeat, while references are uniformly distributed.
            bytes.extend_from_slice(b"mham");
            bytes.extend_from_slice(&(seed as u32).to_le_bytes());
            while bytes.len() < len {
                bytes.extend_from_slice(&record_id(seed.wrapping_add(bytes.len())));
            }
        }
        PayloadKind::MatrixJsonLike => {
            // Repeated Matrix JSON vocabulary is deliberately a positive
            // control: unlike a hash-heavy HAMT node, it should demonstrate
            // the space benefit that motivated per-frame compression.
            const FRAGMENT: &[u8] = br#"{"type":"m.room.message","sender":"@alice:example.org","content":{"msgtype":"m.text","body":"federated event payload with recurring fields"},"origin_server_ts":1710000000000}"#;
            while bytes.len() < len {
                bytes.extend_from_slice(FRAGMENT);
            }
        }
        PayloadKind::Incompressible => {
            let mut state = splitmix64(seed as u64);
            while bytes.len() < len {
                state = splitmix64(state);
                bytes.extend_from_slice(&state.to_le_bytes());
            }
        }
    }
    bytes.truncate(len);
    bytes
}

/// Mirrors the non-zstd work in `packfile::write_record`: v3/v4 record
/// framing, allocation, and CRC, but deliberately stores plaintext payloads.
/// Keeping this here lets the benchmark identify the marginal cost of the
/// current unconditional compressor call without exposing a test-only raw
/// writer through mtxdb's public API.
fn write_record_raw(writer: &mut impl Write, record: &Record) -> io::Result<u64> {
    const FRAME_FIXED_LEN: u32 = 1 + 4 + 16 + 16;

    let uncompressed_len = u32::try_from(record.data.len()).expect("benchmark payload fits u32");
    let frame_len = FRAME_FIXED_LEN
        .checked_add(uncompressed_len)
        .expect("benchmark frame length fits u32");
    let total_len = 4_u64 + u64::from(frame_len) + 4;

    let mut frame = Vec::with_capacity(usize::try_from(total_len).expect("frame fits usize"));
    frame.extend_from_slice(&frame_len.to_le_bytes());
    frame.push(0); // uncompressed flag
    frame.extend_from_slice(&uncompressed_len.to_le_bytes());
    frame.extend_from_slice(&record.collection_id);
    frame.extend_from_slice(&record.hash);
    frame.extend_from_slice(&record.data);
    frame.extend_from_slice(&crc32fast::hash(&frame).to_le_bytes());
    writer.write_all(&frame)?;
    Ok(total_len)
}

struct BlackBoxWriter;

impl Write for BlackBoxWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        std::hint::black_box(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn measure(
    records: &[Record],
    write: impl Fn(&mut BlackBoxWriter, &Record) -> io::Result<u64>,
) -> (Duration, u64) {
    // Warm allocator and codec initialization outside the timed region.
    let mut warm_sink = BlackBoxWriter;
    let _ = write(&mut warm_sink, &records[0]).expect("warm write succeeds");

    let mut sink = BlackBoxWriter;
    let started = Instant::now();
    let bytes = records
        .iter()
        .map(|record| write(&mut sink, record).expect("benchmark write succeeds"))
        .sum();
    (started.elapsed(), bytes)
}

fn run_case(kind: PayloadKind, payload_len: usize) {
    let records: Vec<_> = (0..RECORDS_PER_CASE)
        .map(|index| Record {
            collection_id: COLLECTION_ID,
            hash: record_id(index),
            data: Bytes::from(payload(kind, payload_len, index)),
        })
        .collect();

    let (raw_elapsed, raw_bytes) = measure(&records, write_record_raw);
    let (zstd_elapsed, zstd_bytes) = measure(&records, packfile::write_record);
    let raw_per_record = raw_elapsed.as_secs_f64() * 1_000_000.0 / RECORDS_PER_CASE as f64;
    let zstd_per_record = zstd_elapsed.as_secs_f64() * 1_000_000.0 / RECORDS_PER_CASE as f64;
    let size_ratio = zstd_bytes as f64 / raw_bytes as f64;

    println!(
        "{:<15} {:>5} B  raw {:>8.2} us ({:>9.0} rec/s)  zstd {:>8.2} us ({:>9.0} rec/s)  slowdown {:>5.2}x  stored {:>5.1}%",
        kind.label(),
        payload_len,
        raw_per_record,
        RECORDS_PER_CASE as f64 / raw_elapsed.as_secs_f64(),
        zstd_per_record,
        RECORDS_PER_CASE as f64 / zstd_elapsed.as_secs_f64(),
        zstd_elapsed.as_secs_f64() / raw_elapsed.as_secs_f64(),
        size_ratio * 100.0,
    );
}

fn main() {
    println!(
        "raw vs zstd level-3 packframe production ({} records/case)",
        RECORDS_PER_CASE
    );
    println!("payload type       size       raw                         zstd                         result");
    for kind in PayloadKind::ALL {
        for payload_len in [64, 256, 1_024, 4_096] {
            run_case(kind, payload_len);
        }
    }
}
