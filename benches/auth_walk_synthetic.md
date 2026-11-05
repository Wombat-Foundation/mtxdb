# Synthetic persisted-child versus forward-walk comparison

This benchmark exercises Sithnapse's actual `_fetch_child_event_ids_from_sql`,
`_fetch_and_verify_children` and `get_forward_reachable_short_ids` functions,
extracted from the sibling checkout without importing the homeserver. Its
embedded-store calls use a synthetic SQLite-backed adapter. The alternative is
an equivalent Python traversal of forward auth edges, building a request-local
reverse map as Congruent does. It does not execute Congruent's Rust code.

**These are algorithm/adapter measurements, not end-to-end Sithnapse, Congruent,
or mtxdb timings. They do not establish that a production migration is safe or
faster.** Both methods use the same generated graph, snapshot and point-store
adapter; the persisted path also executes the real helper's SQL verification and
short/event-ID translations. There is no homeserver database, federation, FFI,
asynchronous fetch concurrency, JSON PDU decode, disk I/O, WAL or durability
cost in the measured operations. Instrumentation overhead is included.

## Reproduce

Python 3.10+, the standard library, and `ruff` (for the lint and format checks
below) suffice. The default helper source is
`../sithnapse/synapse/storage/databases/main/embedded_event_auth_chains.py`;
`--sithnapse-source` can override it. Source SHA-256, Python/SQLite versions and
fixture parameters are recorded beside each CSV.

```sh
python3 benches/auth_walk_synthetic.py \
  --output benches/results/auth_walk_synthetic.csv
python3 benches/auth_walk_synthetic.py \
  --sizes 1000,10000 --conflicts 2,8 --incomplete-percent 10 \
  --output benches/results/auth_walk_repair.csv
ruff check benches/auth_walk_synthetic.py
ruff format --check benches/auth_walk_synthetic.py
```

The two benchmark commands should run sequentially. Each case reports three
wall-time samples without memory tracing, as median/min/max. Memory is then
collected as one additional traced sample in the same invocation, outside those
timings. Returned subgraphs are checked against an independent fixture oracle
after every sample. A preliminary fork
fixture checks seed inclusion, pruning of out-of-candidate descendants, SQL
repair of wholly empty embedded child rows, and both discovery modes.

## Fixture and access contracts

- Ancestor union sizes: 1k, 10k, 100k; conflicts: 2, 8, 32.
- A synthetic state-event chain ensures the newest seed has the requested
  ancestor union. Extra piecewise and interleaved references create hubs. No
  create-event auth edge is generated. This is a v12-shaped traversal workload,
  **not a set of protocol-authorized Matrix PDUs**.
- Another union-sized set of children lies outside the ancestor union. They
  appear in stored child lists but must be pruned from the returned subgraph.
- `spanning` seeds start halfway through the union and end at its tip; the
  returned subgraph contains half the union. `tips` seeds are the last C events;
  the returned subgraph contains only those C conflicts. This deliberately
  varies forward-reachable size independently of ancestor-union size.
- `cached_candidates` supplies an existing candidate set, without cached forward
  rows or a reverse map. Persisted traversal can query only reachable parents;
  ephemeral traversal still reads the union to construct its map.
- `discover_candidates` charges both methods for discovering ancestors from
  forward records. The ephemeral method builds its reverse map during that pass;
  the persisted method subsequently queries and SQL-verifies children.
- The complete sweep has no missing inbound entries. The repair sweep omits
  child IDs whose ordinal modulo 100 is below 10; SQL remains authoritative.
  Each sample resets repaired rows outside the timer so later samples do not
  benefit from the first sample's repair. Event bodies/forward rows themselves
  are present; this does not model unavailable forward events.

## Results on this host

Tables below select eight conflicts from the complete sweep; CSV files include
all **104 cases** (72 complete, 32 repair), each with three timed samples and a
separate memory sample. Both methods returned the same expected subgraph in
every sample. Timings are warm in-memory SQLite results, with no isolated CPU or
statistical confidence interval.

| Union nodes | Seeds    | Candidates | Persisted verified ms | Ephemeral forward ms |
| ----------: | -------- | ---------- | --------------------: | -------------------: |
|        1000 | spanning | cached     |                 5.403 |                2.496 |
|        1000 | spanning | discover   |                 7.670 |                2.622 |
|        1000 | tips     | cached     |                 0.070 |                2.343 |
|        1000 | tips     | discover   |                 2.266 |                2.496 |
|       10000 | spanning | cached     |                59.183 |               26.944 |
|       10000 | spanning | discover   |                81.909 |               28.802 |
|       10000 | tips     | cached     |                 0.072 |               25.855 |
|       10000 | tips     | discover   |                23.481 |               27.296 |
|      100000 | spanning | cached     |               637.152 |              325.039 |
|      100000 | spanning | discover   |               867.514 |              307.607 |
|      100000 | tips     | cached     |                 0.076 |              281.813 |
|      100000 | tips     | discover   |               240.101 |              292.355 |

For the 100k-union/eight-conflict case:

| Seeds / candidates  | Returned payload bytes, persisted / ephemeral | Peak new Python bytes, persisted / ephemeral |
| ------------------- | --------------------------------------------: | -------------------------------------------: |
| spanning / discover |                        13,763,744 / 1,196,608 |                      13,720,193 / 26,971,688 |
| tips / cached       |                             1,132 / 1,196,608 |                           5,168 / 19,872,392 |

The spanning/full-discovery case required 100k forward reads for either method;
the persisted path additionally made 50k embedded-child reads, 50k SQL child
queries and 249,368 identity probes. For cached tip candidates, persisted lookup
made eight embedded reads and eight SQL queries, while the ephemeral method
still made 100k forward reads. These differences explain why one synthetic case
favors the forward walk and another favors persisted children.

In the 10k/eight-conflict/spanning/full-discovery repair case, persisted
traversal repaired 1,054 child rows and took 89.420 ms median; the forward walk
took 28.649 ms. The repair result remained identical. This tests the helper's
completeness contract; it does not simulate missing forward records.

## Interpretation and measurement limits

The forward walk traded lower returned payload and fewer lookup layers for a
larger request-local reverse map. When candidates were already available and
only a few nodes were forward reachable, the persisted path avoided rebuilding
that map and was much faster in this stand-in. Therefore these results do not
support unconditionally deleting Sithnapse's persisted reverse-auth dependency.
The useful decision variables are ancestor-union size, reachable fraction,
existing caches, child-list completeness and the cost of SQL verification.

`payload_bytes` counts returned adjacency bytes, translated IDs and SQL event-ID
strings; it excludes record headers, pages, indexes and writes. `forward_reads`,
`embedded_reads` and `identity_probes` count logical point probes; `sql_queries`
counts indexed child-list queries. **They are not physical or random disk
reads.** SQLite is in memory, so cold-storage conclusions cannot be drawn.

`python_peak_bytes` is incremental traced Python allocation during the query,
including its result. It excludes fixture/candidate data created before tracing,
SQLite's native allocation, process RSS and any real engine caches/indexes. It
is not a prediction of Congruent's Rust memory use. Identity resolution uses a
resident synthetic dictionary; forward reads return packed u32 edges, whereas a
real event-fetch path may read and parse a larger PDU. Neither limitation should
be turned into a claimed homeserver speedup.

Before migration, use the relevant embedding against generated mtxdb data and
measure its transaction/SQL/FFI costs, memory limits and missing-event
semantics. The retired CLI per-parent writer is independent of this gate and
remains removed. No sidecar run rows, Rezzy pin bump or homeserver migration is
included.
