# Incremental raw-import frontier

## Problem

A raw import reads one file at a time, but check 5 (does the event authorize
against the state before it?) can only be answered from the room's whole DAG.
The current implementation therefore calls `load_collection_events`, rebuilds a
synthetic export from every stored event, and replays the full topological walk
for every file. With `k` files and `n` events the work is `O(k * n)`: the same
stored events are read and judged again and again.

This is acceptable for a single-room triage dump. It is not acceptable when a
large room arrives as many JSONL shards.

## What must stay true

- Check 4 runs before check 5.
- Rejected events never contribute state to later events.
- Missing auth or prev events leave an event unchecked, not accepted.
- A later file may complete a base that an earlier file lacked. That can change
  an earlier event's derived verdict, so an incremental design must be able to
  revisit a bounded set of earlier events.
- Source `rejected`/`soft_failed` claims are evidence for disagreement reports
  only. They never decide state or a derived verdict.
- Raw auth_chain-only events are judged but are not stored, so they must not
  acquire verdicts, flags, or frontier entries of their own.

## Persisted frontier

Store one versioned record per raw collection, in a new auxiliary namespace
keyed by the collection id. The payload should contain:

- `room_id`, `room_version`, and the template / collection identity.
- `heads`: event ids that are still DAG leaves for this collection.
- `unresolved`: events that could not be judged because a parent or auth event
  was absent. Each entry carries the missing ids and the set of state groups
  that are already known for its parents.
- `dirty`: earlier events whose stored verdict may change when `unresolved`
  is completed. This is normally the unresolved events plus descendants that
  have already been stored.
- `claims`: the source `rejected` / `soft_failed` / rejection-code claims for
  stored events, so redaction cannot erase the evidence used for disagreement
  reporting. This also preserves a soft-fail claim on an event that was later
  rejected: the claim is separate from the derived verdict.
- `state_at_heads`: state-group ids (or HAMT roots) for each head, so a new
  event whose parent is a head does not need to rebuild state from the room
  root.
- A format version and a checksum.

The record must be written in the same transaction as the event batches whose
frontier it describes. If the transaction fails, the old frontier remains
valid.

## Algorithm for a new file

1. Read the frontier record. If it is absent or does not verify, fall back to
   the current full scan and rebuild it on success.
2. Parse the new events. Build the set of ids referenced by their `prev_events`
   and `auth_events`.
3. Load only those referenced events from the collection, plus any frontier
   `unresolved` entries whose missing ids are now present. Do not scan the
   whole collection.
4. Topologically order the new events and the affected `dirty`/`unresolved`
   events. Events whose parents are still absent stay in `unresolved`; they
   are not accepted just because this file has no answer for them.
5. For each event with a parent in `state_at_heads`, start from that state
   group. For multiple parents, state-resolve the parent state groups using
   the existing state-resolution path. If no parent state is available, leave
   the event unchecked.
6. Run check 4, then check 5 on that bounded subgraph, applying the same
   cascade and rejection rules as the full walk. A newly available parent can
   move an event from `unresolved`/accepted to rejected, or remove a forced
   rejection that came from a missing auth event.
7. Write verdicts and flags only for events whose derived result changed. The
   writer already compares the stored verdict; flag staging should likewise
   skip chunks whose bits did not change.
8. Replace the frontier with:
   - the new DAG leaves,
   - the still-unresolved events,
   - their dirty descendants,
   - the state groups needed by the new leaves,
   - the accumulated source claims for stored events.

## Why this is not just a cache

The frontier is the minimal correctness boundary. Everything reachable only
through a head is already judged and cannot change unless a new event connects
to it. The `unresolved` and `dirty` sets are the only earlier events allowed to
be revisited. If a batch cannot be connected to any known head, the importer
falls back to a full replay for that batch rather than guessing.

## Cost

For a batch of `m` new events, the walk touches `O(m + u + d)` events, where
`u` is the unresolved boundary and `d` its dirty descendants, instead of
`O(n)`. State reads are limited to the state groups named by the involved
parents. The frontier record itself is bounded by the number of DAG leaves and
unresolved events, not by the room's event count.

## Rollout and tests

- Keep the full-scan path as the fallback and as the reference implementation.
  An incremental run can optionally compare its verdicts with a full replay
  under a test-only flag.
- Tests should cover:
  - several files in order, including a late parent that changes an earlier
    event from unchecked to rejected;
  - a missing auth event that arrives later and removes a forced rejection;
  - rejected state not contributing to later state;
  - soft-fail across files and a rejected event that also carried a soft-fail
    claim;
  - disagreement warnings restricted to the current file;
  - a reimport that changes nothing and appends no verdict, flag, or auth
    counter records;
  - an out-of-order file that cannot be connected, which must fall back rather
    than accept events unchecked.
- A rebuild command can discard the frontier and reconstruct it with one full
  scan. That is also the recovery path after a corrupt record.
