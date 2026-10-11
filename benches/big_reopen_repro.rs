//! Reproduction: does a store with N tiny records find them all after one clean reopen?
//!
//! Builds N records (`MTXDB_BR_RECORDS`, default 17,602,258), `sync_all`, drops,
//! reopens once, and probes every `MTXDB_BR_STRIDE`th sequence number. Prints the
//! open path and the first missing sequence number. Nothing else touches the
//! store; env knobs `MTXDB_BR_CYCLES`, `MTXDB_BR_EVICT`, `MTXDB_BR_FOREIGN_FILE` repeat the
//! reopen with eviction and a foreign file in the directory.

use std::path::PathBuf;
use std::time::Instant;

use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::PackfileStorage;

const COLLECTION: [u8; 16] = [0xE7; 16];

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn node_id(seq: u64) -> NodeId {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&mix(seq).to_le_bytes());
    id[8..].copy_from_slice(&mix(seq ^ 0xabcd_ef01).to_le_bytes());
    id
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let records = env_u64("MTXDB_BR_RECORDS", 17_602_258);
    let stride = env_u64("MTXDB_BR_STRIDE", 997).max(1);
    let base = std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
    let dir = base.join(format!("mtxdb_bench_big_reopen_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let started = Instant::now();
    let store = PackfileStorage::open(dir.clone()).expect("open new store");
    let data = NodeData::from_slice(&[0x5a; 16]);
    for seq in 0..records {
        store.put(&COLLECTION, &node_id(seq), &data).expect("put");
    }
    store.sync_all().expect("sync_all");
    drop(store);
    println!(
        "built {records} records in {:.1} s",
        started.elapsed().as_secs_f64()
    );

    // The failing run also left a foreign `pack.idx` in the store directory and
    // evicted the page cache between opens; both are switchable here.
    if env_u64("MTXDB_BR_FOREIGN_FILE", 1) == 1 {
        std::fs::write(dir.join("pack.idx"), b"MTXIDX01").expect("write foreign file");
    }
    if env_u64("MTXDB_BR_DROP_CHECKPOINT", 0) == 1 {
        // Force the startup FullScan path.
        for entry in std::fs::read_dir(&dir).expect("ls").flatten() {
            if entry.file_name().to_string_lossy().starts_with("index.") {
                std::fs::remove_file(entry.path()).expect("remove index file");
            }
        }
    }
    let cycles = env_u64("MTXDB_BR_CYCLES", 4);
    let evict = env_u64("MTXDB_BR_EVICT", 1) == 1;
    let mut missed = 0u64;
    for cycle in 1..=cycles {
        if evict {
            let _ = std::process::Command::new("sync").status();
            let _ = std::process::Command::new("vmtouch")
                .arg("-e")
                .arg(&dir)
                .output();
        }
        let store = PackfileStorage::open(dir.clone()).expect("reopen");
        let timings = store.open_timings().expect("timings");
        let (mut probed, mut cycle_missed, mut first_missing) = (0u64, 0u64, None);
        let mut seq = 0;
        while seq < records {
            probed = probed.checked_add(1).expect("probe count overflow");
            if store
                .get(&COLLECTION, &node_id(seq))
                .expect("get")
                .is_none()
            {
                cycle_missed = cycle_missed
                    .checked_add(1)
                    .expect("cycle miss count overflow");
                first_missing.get_or_insert(seq);
            }
            seq = seq.checked_add(stride).expect("probe sequence overflow");
        }
        println!(
            "cycle {cycle}: path {:?}, open {:?}, probed {probed}, missed {cycle_missed}, first missing {first_missing:?}",
            timings.path, timings.total
        );
        missed = missed
            .checked_add(cycle_missed)
            .expect("total miss count overflow");
        drop(store);
    }
    if missed == 0 {
        let _ = std::fs::remove_dir_all(&dir);
    } else {
        println!("kept {} for inspection", dir.display());
    }
}
