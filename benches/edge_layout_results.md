# Chunked edge layout: storage, lookup and write cost

Benchmark-only comparison of the current `ShortIdIndex` edge store (the
**baseline**) against a **chunked prototype** layout, on one real Matrix room.
Nothing here is a production format.

Scope: the core layout and lookup tables use one room (77,454 events, 77,653
ids), one machine, and 20,000 sampled operations per lookup measurement, with
no concurrency. The later sithnapse-shaped write comparison uses a different
room and is labeled separately below. Rewritten-record garbage is measured by
simulated compaction where stated; the engine's own compaction path is not
measured. All figures are workload- and hardware-specific.

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

## Small-batch writes and garbage

Events arrive in batches of B, appended after a 60,000-event preload (batches
of 1,000), chunk 32, zstd + CRC, one machine, HDD. Fixture: the aria-net.org v6
room (75,377 events; `docs/edge-layout-room.tsv`, from `benches/room_to_tsv.sh`).
It is **not the room used above**, so these numbers do not compare with the
tables above. Sync every 20 batches (the engine does not fsync per commit); the
wall time includes those syncs, which is most of it at batch 1 and 2.

| batch | baseline us/event | prototype us/event | baseline growth B/event | prototype growth B/event | prototype rewrites |
| ----- | ----------------- | ------------------ | ----------------------- | ------------------------ | ------------------ |
| 1     | 2,453-2,572       | 2,588-2,868        | 930                     | 1,320                    | 97%                |
| 2     | 1,223-1,314       | 1,245-1,269        | 841                     | 689                      | 96%                |
| 10    | 276-321           | 238-263            | 770                     | 200                      | 83%                |

(2 reps; ranges.) Wall time is within noise at batch 1 and 2: about 5 of the
roughly 5-6 s at batch 1 is the 100 periodic fsyncs. On disk the prototype
grows 42% more at batch 1, and less from batch 2 up.

Garbage (1 rep, the same runs): dead share is the part of pack bytes that
a **simulated** compaction removes (copy only the newest version of each record
into a fresh store; not the engine's compaction path).

| batch | packs    | after compaction | dead | copy time |
| ----- | -------- | ---------------- | ---- | --------- |
| 1     | 6.12 MiB | 4.33 MiB         | 29%  | 485 ms    |
| 2     | 5.17 MiB | 4.27 MiB         | 17%  | 575 ms    |
| 10    | 4.43 MiB | 4.21 MiB         | 5%   | 435 ms    |
| 100   | 4.27 MiB | 4.20 MiB         | 2%   | 479 ms    |

Baseline garbage is not measured. With a sync after **every** batch on the
earlier room (1 rep), batch 1 took 33.5 ms/event (baseline) vs 39.8 ms
(prototype): that run measured fsync latency, not chunk rewrites, and is a worst
case.

## Sithnapse-shaped bidirectional auth writes

This follow-up compares the modeled record-per-list layout (outbound auth list
per event plus read-modify-write child lists per auth parent) with the chunked
prototype storing both directions. It uses the aria-net.org v6 room fixture
(75,377 events, 77,653 distinct ids), not the room used by the preceding
tables. Each case preloads 60,000 events in batches of 1,000, then appends
2,000 events at the stated batch size, syncing every 20 batches. These are
single-run results. The replay uses the same mtxdb `Database` settings but is
not sithnapse's engine or full ingest path: exact node-id derivation, SQL work,
short-id lookup cost, and closure-cache behavior are outside this measurement.

| batch | record layout µs/event | chunked µs/event | record growth B/event | chunked growth B/event | record payload B/event | chunked payload B/event |
| ----- | ----------------------: | --------------: | --------------------: | ---------------------: | ---------------------: | ----------------------: |
| 1     | 13,595                  | 16,600          | 167,107               | 177,765                | 275,201                | 361,040                 |
| 2     | 11,491                  | 11,244          | 85,638                | 88,974                 | 142,095                | 180,731                 |
| 10    | 1,959                   | 1,977           | 17,684                | 17,817                 | 31,053                 | 36,375                  |
| 100   | 172                     | 109             | 2,750                 | 1,921                  | 3,635                  | 3,786                   |

The wall times include periodic syncs. Pack sizes below are for the entire
preload-plus-append store; "after simulated compaction" copies the newest
payload per record id into a fresh store and is not the engine's compaction
implementation.

| batch | record-layout packs | record after compaction | record dead | chunked packs | chunked after compaction | chunked dead |
| ----- | ------------------: | ----------------------: | ----------: | ------------: | -----------------------: | -----------: |
| 1     | 347.35 MiB          | 17.66 MiB               | 95%         | 351.17 MiB    | 4.89 MiB                 | 99%          |
| 2     | 191.91 MiB          | 17.66 MiB               | 91%         | 181.80 MiB    | 4.82 MiB                 | 97%          |
| 10    | 62.17 MiB           | 17.66 MiB               | 72%         | 46.27 MiB     | 4.77 MiB                 | 90%          |
| 100   | 30.09 MiB           | 17.66 MiB               | 41%         | 15.64 MiB     | 4.76 MiB                 | 70%          |

This does **not** show a chunked win for one-event-at-a-time ingest: at batch 1
it is slower and grows slightly more; batches 2 and 10 are effectively tied in
wall time and have similar growth. At batch 100, chunking is faster and has
less growth. In both layouts, hot auth parents make the child list grow with
the room and force repeated whole-list writes. The chunked prototype still
rewrites the child chunk containing that parent on each append, so chunking
does not solve the hot-parent write amplification. The measured 242 KiB
hot-parent child list fits the prototype child-chunk encoding: child counts and
offsets are `u32`; the separate `u8` lengths in the edge chunk apply to
outbound per-event lists, whose maximum auth-list length in this fixture is 5.

These results point toward segmented or append-friendly child lists (for
example, immutable child segments plus a small delta), not rewriting one
ever-growing parent list. That is a design direction, not implemented here.

## Sithnapse's two-direction auth edges

Sithnapse does not use `ShortIdIndex`. `edge_layout_sithnapse.rs` replays its
write shape on the aria-net v6 room (60,000 preloaded, 2,000 appended, sync every
20 batches, 1 rep, auth adjacency only): per event an outbound auth record and the
id records, plus a read-modify-write of the whole child-list record of every auth
parent. It is a replay through `Database`, not sithnapse's engine or node ids. The
chunked column stores the same two directions in chunk 32 records.

| batch | layout     | us/event | growth B/event | dead packs | compacted packs |
| ----- | ---------- | -------- | -------------- | ---------- | --------------- |
| 1     | sithnapse  | 16,383   | 167,107        | 95%        | 17.66 MiB       |
| 1     | chunked    | 15,041   | 177,765        | 99%        | 4.89 MiB        |
| 2     | sithnapse  | 10,342   | 85,638         | 91%        | 17.66 MiB       |
| 2     | chunked    | 8,462    | 88,974         | 97%        | 4.82 MiB        |
| 10    | sithnapse  | 2,358    | 17,684         | 72%        | 17.66 MiB       |
| 10    | chunked    | 1,546    | 17,817         | 90%        | 4.77 MiB        |
| 100   | sithnapse  | 154      | 2,750          | 41%        | 17.66 MiB       |
| 100   | chunked    | 131      | 1,921          | 70%        | 4.76 MiB        |

(chunked batch 1 and 100 wall times are from a second run; the first run gave
19,650 and 130.7 us/event, so single runs vary by about 30% at batch 1.)

The hot parents (create, power levels) are auth parents of nearly every event, so
their child lists grow with the room (largest record 242 KiB) and are rewritten
whole on every append. That is the dominant cost in **both** layouts: growth is
about 170 KB per event at batch 1, and 95-99% of pack bytes are dead versions.
Chunking does not fix it. It does shrink the compacted store 3.6x. A fix needs an
append-friendly child layout, which is a design decision.

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
- Real compaction cost and scheduling; the reported compaction is a copy-based
  simulation, not the engine's reclaim path.
- Concurrent readers and writers.
- Rooms larger than 77k events.
- Real sithnapse ingest-path cost beyond the modeled records (SQL fetch,
  resolves, engine integration, and closure-cache effects).
- Collision-safe production lookup and atomic publication of a chunked index.
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
