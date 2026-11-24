# mtxdb

A "write-once" `packfile` storage with optimal layout and binary format.

Running Matrix servers on spinning, mechanical drives—made _less_ painful.

- Disk offsets probed in average-case `O(1)` time via open-addressed
  linear probing.
- Binary serialization format repacked periodically in topological order, making
  nominal C2S and federation DB reads purely sequential.
- Stores edge and inverted edge indexes, facilitating near pure sequential
  segments.
- Records are never rewritten in place. Removing a key appends a versioned
  removal marker (one-way format addition: older builds refuse to open a store
  that holds one); space returns at repack.

## Collection templates

The storage engine is format-agnostic. Its application-specific mapping policy
is declared by a versioned JSON collection template: record identity, collection
key, payload-retention policy, relationships, and validation behaviour. The Matrix
reference policy is [`templates/matrix-event-v1.json`](templates/matrix-event-v1.json);
see [`templates/README.md`](templates/README.md) for the generic format and why
Matrix uses `event_id` as its logical identity.

## On-disk diagnostics

The CLI's read-only metadata inspector reports physical storage and durability
state without acquiring writer locks:

```text
mtxdb meta [overview|db|wal|checkpoint|delta|sidecars|packs|locks|raw]
```

The default target is `overview`. Use `--limit` and `--offset` for pagination,
`--decode` for supported payload decoding, and `--json` for structured output.
Corrupt headers, truncated tails, stale checkpoints, orphaned delta epochs, and
lock-marker anomalies are reported as diagnostics while inspection continues
where possible.

The process-safety policy for read-only tools and the `multi-reader` feature is
documented in
[`docs/docs/2026-09-30-process-safety-and-reader-builds.md`](docs/docs/2026-09-30-process-safety-and-reader-builds.md).

## CLI layout

The CLI groups Matrix room operations under `room` and pack operations under
`packs`. `info` accepts a collection, room, or pack selector; use it for the
summary view of any one of those objects.

```text
mtxdb packs list [--layout]
mtxdb packs scan <PACK_ID> [--collection <COLLECTION>]
mtxdb packs dump --pack <PACK_ID> [--collection <COLLECTION>] [--out <FILE>]
mtxdb packs extract --pack <PACK_ID> --collection <COLLECTION> --out <FILE>
mtxdb scan <COLLECTION>
mtxdb info [<SELECTOR>] [--stats] [--tombstones [--detail] [--keys]]
mtxdb room timeline <ROOM_ID>
mtxdb room flags <ROOM_ID>
mtxdb room members <ROOM_ID>
mtxdb room check <ROOM_ID> [--auth] [--prevs] [--status <STATUS>]
mtxdb room gaps <ROOM_ID> {ingest|waive|anchor}
mtxdb get [--collection <COLLECTION>] <ID>
mtxdb remove <COLLECTION> <ID>...
mtxdb drop <COLLECTION>...
```

`<SELECTOR>` for `info` is a collection ID, a canonical room ID, or a pack ID.
With neither `--auth` nor `--prevs`, `room check` runs both audits; `--status`
filters the prev-event gaps section, which also runs by default. `get` takes its
record ID positionally; the former `-i/--id` spelling is not supported.
The `room check --auth` audit reads the auth graph through the database's shared
WAL, so it requires a usable shared WAL at the database root.
