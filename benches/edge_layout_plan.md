# Chunked edge layout: measurement plan

Benchmark-only. Nothing here touches `src/`. Each stage runs the baseline
(`ShortIdIndex`) next to the prototype on the same room and events, and ends
with a pass/fail line. A stage that fails stops the sequence until the design
is revisited. Thresholds are proposals; change them before running, not after.

Common: room `MTXDB_EL_TSV` (77,454 events), chunk 32, zstd + CRC (default
policies), 3 repetitions, ranges reported, one machine, no concurrency.

## Stage 1: small-batch writes

Question: does the chunk rewrite cost stay acceptable when events arrive one or
two at a time?

Protocol: preload the first 60,000 events in batches of 1,000 (both layouts),
then append the next 2,000 events in batches of B = 1, 2, 10, 100, with a sync
after every batch (each arrival durable). Fresh store per B.

Report per event: wall time, prototype payload bytes put, on-disk growth
(packs + index), and the prototype's share of payload that rewrites an earlier
chunk. Baseline growth is read the same way.

Pass: at every B, prototype growth per event is at most the baseline's, and
prototype wall per event is at most 1.5x the baseline's.
Fail: either exceeds. Then the design needs a delta/WAL front (flush at 32
entries or on demand) and Stage 1 repeats against that.

## Stage 2: garbage and compaction

Question: how much of the store is dead chunk versions, and what does reclaiming
it cost?

Protocol: after each Stage 1 run, count live bytes (latest version of each
record id) against total pack bytes. Then simulate compaction: copy the live
records into a fresh `Database`, time it, and compare final size to the
baseline's size for the same events. A forced baseline checkpoint is the
baseline compaction.

Pass: garbage is at most 2x live bytes at B = 1 and 2 (or compaction brings the
store back within 10% of its freshly written size) and compaction time is
linear in live bytes.
Fail: otherwise; chunk rewrite needs a smaller unit or the delta front from
Stage 1.

## Stage 3: real adjacency query paths

Question: what do the queries the engine actually serves cost?

Protocol: run the baseline `MatrixAdjacency` `short_id`, `prev_of`, `auth_of`,
`relation_of` and `RoomAuth::auth_chain_ids` on the room. For the prototype,
emulate each in the bench from the chunk records with the same call shape
(key -> id with key confirmation, edge-list chunk read, one batched id -> key
resolve over the targets that fetches each key chunk once). Warm; cold only when
`evict_cold` accepts the store (mmapped packs resist eviction, so cold is
reported as inconclusive otherwise).

Pass: `auth_of`/`prev_of` at most 2x baseline per call and `auth_chain_ids` at
most 1.5x, both against the same events.
Fail: otherwise; the read cost outweighs the storage and write gains and the
prototype is not adopted as is.

## Stage 4: collision-safe lookup (injected hash)

Question: does the hash run plus full-key confirmation plus collision bucket
stay correct when hashes collide?

Protocol: a synthetic test with a hash function injected into the format
helpers so chosen keys collide. Cases: two keys to one hash (bucket promotion),
resolve both, a third key with the same hash that was never inserted (miss),
a key whose hash matches nothing, a bucket read after reopen, and an update that
fails between the chunk write and the run write (reader must see the old state
or an error, never a wrong id). Then the cost of confirmation on the room.

Pass: every case returns the exact id or a miss or an error, never another
key's id; confirmation cost is reported with Stage 3's key -> id numbers.
Fail: any wrong id.

## Out of scope until Stages 1-4 pass

Concurrency, rooms larger than 77k events, other arrival orders, production
wiring, the `ShortIdIndex` API.
