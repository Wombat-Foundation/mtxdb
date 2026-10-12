# Chunked edge layout: storage, lookup and write cost

Benchmark-only comparison of the current `ShortIdIndex` edge store (the
**baseline**) against a **chunked prototype** layout, on one real Matrix room.
Nothing here is a production format.

Scope, stated once: **one room** (77,454 events, 77,653 ids), **one machine**,
20,000 sampled operations per measurement, **no concurrency**, and compaction
of the prototype's rewritten chunks **not measured**. Numbers are workload- and
hardware-specific.

## Storage size

The prototype stores owners and edge lists together; the baseline stores them
as separate records per edge family.

| ids per chunk | prototype size | B/event | baseline size | B/event |
| ------------- | -------------- | ------- | ------------- | ------- |
| 32 (zstd)     | 6.30 MiB       | 85.2    | 41.52 MiB     | 562.1   |
| 256 (zstd)    | 8.11 MiB       | 109.7   | 41.52 MiB     | 562.1   |
| 32 (raw)      | 8.42 MiB       | 114.4   | 41.52 MiB     | 562.1   |
| 256 (raw)     | 14.78 MiB      | 201.9   | 41.52 MiB     | 562.1   |

At 32 ids with zstd the prototype is **6.6× smaller** than the baseline.
Compression accounts for a significant part of that: the same layout without
zstd is 8.42 MiB (32 ids) and 14.78 MiB (256 ids), so zstd shrinks the store
by 25–45%.

## Write cost

| ids per chunk | prototype wall | payload written | rewrites |
| ------------- | -------------- | --------------- | -------- |
| 32            | 0.68 s         | 7.80 MiB        | 18%      |
| 256           | 0.84 s         | 14.61 MiB       | 60%      |

The baseline writes the same events in 3.3–3.4 s. The prototype is **4–5×
faster on writes** but pays for it in rewrite amplification: at 256 ids, 60%
of the payload is chunk rewrites (garbage until compaction, which is not
measured here).

## Warm read split

This isolates where one id→key lookup's time goes, under each combination of
the pool's two per-byte policies (zstd compression, CRC32 verification).
Layers are measured independently, not subtracted. Validation sits outside the
timed region.

### 32 ids per chunk (1.5 KB decoded record)

| layer                              | time (µs, 3 reps) |
| ---------------------------------- | ----------------- |
| decode only (`key_at`)             | 0.02–0.04         |
| fetch, no compression, no CRC      | 0.77–0.85         |
| fetch, no compression, CRC on read | 0.60–0.84         |
| fetch, zstd, no CRC                | 4.79–5.07         |
| fetch, zstd, CRC on read (default) | 4.47–4.99         |
| baseline resolve (floor)           | 1.60–2.16         |

### 256 ids per chunk (12 KB decoded record)

| layer                              | time (µs, 3 reps) |
| ---------------------------------- | ----------------- |
| decode only (`key_at`)             | 0.02–0.03         |
| fetch, no compression, no CRC      | 0.44–0.47         |
| fetch, no compression, CRC on read | 1.19–2.19         |
| fetch, zstd, no CRC                | 13.66–14.23       |
| fetch, zstd, CRC on read (default) | 13.92–14.23       |
| baseline resolve (floor)           | 1.51–2.11         |

### What the warm split shows

**zstd decompression dominates.** Fetching the same record without compression
takes ~0.45 µs at both chunk sizes; with zstd it takes 4.7 µs (32 ids) and
14 µs (256 ids). That accounts for essentially all of the per-byte cost
observed in the chunk-size sweep.

**Decode is negligible.** The offset-table `key_at` takes 20–40 ns regardless
of chunk size.

**CRC's contribution is smaller and noisier.** The no-compression CRC timings
vary (0.60–0.84 µs at 32 ids, 1.19–2.19 µs at 256 ids), and the
zstd/no-CRC versus zstd+CRC difference is within noise. The exact CRC cost is
not isolated precisely; it appears to be on the order of 0.4–1 µs.

**The fetch timing covers** `Database::edges().get(...)` only, which includes
the index lookup, pack read, optional CRC verification, optional zstd
decompression, and a copy into `Bytes`. Each layer is timed independently.

## Cold reads

Cold = reopen the database, then `sync` + `vmtouch -e` to evict the store's
files from the page cache, then time the first `MTXDB_EL_COLD` calls.
The packs are read through `memmap2` (`src/engine/shard.rs`), and pages mapped
by a live process resist `vmtouch -e`, so `drop_caches` may not fix this either.
The harness retries eviction up to 5 times and skips any run whose residency
remains above 5%, so a failed eviction produces `SKIPPED` instead of a
warm read mislabeled as cold.

### Eviction results

| store                     | resident after eviction (32 ids) | resident after eviction (256 ids) |
| ------------------------- | -------------------------------- | --------------------------------- |
| baseline                  | 28.9% → SKIPPED                  | 28.9% → SKIPPED                   |
| prototype, zstd           | 3.1% → valid                     | 0.4% → valid                      |
| prototype, no compression | 98.9% → SKIPPED                  | 56.3% → SKIPPED                   |
| proto-merged / hash-only  | 77.9% → SKIPPED                  | 77.9% → SKIPPED                   |

Only the zstd prototype store evicts cleanly. The uncompressed prototype,
the baseline, and the proto-merged/hash-only stores resist eviction and their
cold numbers are skipped.

### Valid cold numbers (zstd stores only)

**32 ids per chunk, zstd store, 3.1% resident after eviction:**

| layer          | cold (µs, 3 reps) | warm (µs) |
| -------------- | ----------------- | --------- |
| fetch only     | 140–382           | 4.5–5.0   |
| fetch + decode | 142–215           | 4.7–5.2   |

**256 ids per chunk, zstd store, 0.4% resident after eviction:**

| layer          | cold (µs, 3 reps) | warm (µs) |
| -------------- | ----------------- | --------- |
| fetch only     | 190–335           | 13.7–14.2 |
| fetch + decode | 217–322           | 14.0–14.6 |

The zstd prototype's cold fetch is 30–55× its warm fetch, reflecting disk I/O
adding ~150–330 µs per lookup. The variance across reps is high (one rep at
32 ids was 140 µs, another 382 µs), consistent with page-cache warmth varying
between runs on this HDD.

## Lookup latency (full prototype, zstd + CRC)

Warm single calls, µs/op, by ids per chunk. 32 is the range of 3 runs; 256 is
one run. Cold is not reported: the stores are mmapped (see Cold reads), and
only the zstd prototype's evicted cleanly.

| operation                   | 32 ids    | 256 ids | baseline  |
| --------------------------- | --------- | ------- | --------- |
| key → id hit (confirmed)    | 5.2–5.5   | 14–16   | 1.45–1.83 |
| key → id miss               | 0.29–0.33 | -       | 0.90      |
| id → key                    | 4.6–4.8   | 14–16   | 1.4–1.6   |
| edge lists (all 3 families) | 5.0–5.4   | 13.8    | 3.7–4.2   |
| key → id hit (hash-only)\*  | 0.33–0.41 | -       | -         |

\*hash-only is **not collision-safe**; it prices the confirming read only.

At 32 ids per chunk the prototype's confirmed hits and id → key are about
3–3.5× the baseline, edge lists about 1.3×, and misses about 3× faster. At 256
ids the point reads are 8–9× the baseline and edge lists 3.3×.

## What this does and does not show

**Shown:**

- The prototype is 6.6× smaller on disk (32 ids, zstd) and 4–5× faster to write.
- Warm id→key latency is dominated by zstd decompression, not decode or CRC.
- The offset-table decode is negligible (20–40 ns).
- At 32 ids the warm point reads are 3–3.5× the baseline and edge lists 1.3×; at 256 ids 8–9× and 3.3×.
- On cold reads (zstd store only), the prototype's cold fetch is 30–55× warm,
  adding 150–330 µs per lookup.

**Not shown / open:**

- Cold reads for the uncompressed prototype, baseline, and merged/hash-only
  stores (eviction failed, skipped by the harness).
- Garbage from rewritten chunks until compaction.
- Concurrent readers and writers.
- Rooms larger than 77k events.
- Whether disabling zstd improves real lookup performance (the warm data says
  it would; the cold data cannot confirm because the uncompressed store's
  eviction is broken).

## Reproducing

```bash
MTXDB_EL_TSV=room.tsv MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
  cargo +1.90.0 bench --manifest-path benches/Cargo.toml --bench edge_layout_lookup

MTXDB_EL_CHUNK=32 MTXDB_EL_TSV=room.tsv \
  MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
  cargo +1.90.0 bench --manifest-path benches/Cargo.toml --bench edge_layout_read_split
```

Chunk-format boundary tests (synthetic, no room data needed):

```bash
cargo +1.90.0 test --manifest-path benches/Cargo.toml --test edge_layout_format
```
