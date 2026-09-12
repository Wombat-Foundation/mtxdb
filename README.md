# mtxdb

A "write-once" `packfile` storage with optimal layout and binary format.

Running Matrix servers on spinning, mechanical drives—made _less_ painful.

- Disk offsets probed in average-case `O(1)` time via open-addressed
  linear probing.
- Binary serialization format repacked periodically in topological order, making
  nominal C2S and federation DB reads purely sequential.
- Stores edge and inverted edge indexes, facilitating near pure sequential
  segments.

## Collection templates

The storage engine is format-agnostic. Its application-specific mapping policy
is declared by a versioned YAML collection template: record identity, collection
key, payload-retention policy, relationships, and validation behaviour. The Matrix
reference policy is [`templates/matrix-event-v1.json`](templates/matrix-event-v1.json);
see [`templates/README.md`](templates/README.md) for the generic format and why
Matrix uses `event_id` as its logical identity.
