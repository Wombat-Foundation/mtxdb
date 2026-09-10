# Collection templates

A collection template declares how an application maps source records into the
generic mtxdb storage model. It is intentionally declarative: the storage core
only stores a 16-byte collection ID, a 16-byte node ID, and opaque bytes. A
template gives an importer enough information to derive those IDs and to state
which source bytes are retained.

`format: mtxdb.collection-template/v1` is the current format. Its generic
reference grammar is [`collection-template-v1.yaml`](collection-template-v1.yaml).
[`matrix-event-v1.yaml`](matrix-event-v1.yaml) is one application profile; it
does not define the generic vocabulary.

## Semantics

| Section                 | Meaning                                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------------------ |
| `input`                 | Accepted source encoding and record framing.                                                           |
| `record.identity`       | JSON field used as the logical record identity and the stable digest used for mtxdb's 16-byte node ID. |
| `record.payload`        | Complete source retention or an explicit derived projection.                                           |
| `collection.key`        | JSON field that scopes nodes into a collection and the digest used for its 16-byte collection ID.      |
| `collection.display_id` | User-facing identifier, distinct from the internal digest key.                                         |
| `metadata`              | Application metadata persisted with the collection.                                                    |
| `primordial`            | The record that establishes a collection's metadata and invariants before it may be created.           |
| `relationships`         | Optional reference fields used for graph traversal; they do not change the stored object.              |
| `validation`            | Required fields and the defined behaviour for records that do not meet the template.                   |
| `extensions`            | Namespaced application policy; the generic storage engine does not interpret it.                       |

JSON paths are RFC 6901 JSON Pointers. `"/event_id"` therefore identifies a
top-level `event_id`; `"/"` means the complete source object.

## Retention policy

`record.payload.mode: source` is the safe default. It preserves unknown fields, signatures,
hashes, `unsigned`, and future protocol extensions. It is appropriate whenever
the database is an archive or an interchange format.

`record.payload.mode: projection` is opt-in and requires an explicit `include` list. It is
for derived caches only: omitting a field is a data-model decision and must not
be presented as a lossless import. A template must never use an implicit
deny-list or silently discard fields it does not recognize.

## Collection metadata and primordial records

The 16-byte collection ID in a pack frame is an internal digest key, never the
collection's user-facing name. `collection.display_id` declares the source
identifier shown by user interfaces and exports. `metadata` is persisted
with the collection rather than inferred from arbitrary later records.

Templates that need collection-wide invariants declare a `primordial` rule. A
new collection is admitted only after one record matches that rule and its
metadata can be extracted. Subsequent partial imports may use existing,
validated metadata, but cannot create a new collection without the primordial
record. This avoids accepting a disconnected fragment as if it described a
complete collection.

For Matrix, the primordial record is the `m.room.create` state event with an
empty `state_key`. Room-version-12 create PDUs can omit `room_id`, so membership
is established from another room record's `auth_events` reference to the create
event, rather than requiring `room_id` on the create PDU itself.

## Versioned protocol behaviour

An application extension may bind a collection to a version-specific policy
when its primordial record is accepted. This is necessary when a record's
identity rules vary by protocol version. The Matrix extension does so through
the executable `MatrixRoomVersion` policy:

| Room version | Event ID                                          | State resolution | Canonical-number rule | Storage impact                                                                                         |
| ------------ | ------------------------------------------------- | ---------------- | --------------------- | ------------------------------------------------------------------------------------------------------ |
| 1            | Server-assigned                                   | V1               | Lenient               | Store the declared `event_id`; it cannot be recomputed from the event.                                 |
| 2            | Server-assigned                                   | V2               | Lenient               | Same identity treatment; only the state resolver changes.                                              |
| 3            | SHA-256 reference hash of canonical redacted JSON | V2               | Lenient               | A verified import must compute the version-3 redacted canonical preimage and compare it to `event_id`. |

Room versions 1–5 share the original redaction table. Version 6 introduces
strict canonical-number validation; later room-version changes have their own
profile entries when the implementation supports them.

The executable Rust policy currently covers numeric room versions 1–12. The
additional storage-relevant boundaries are: v4 switches reference IDs to
URL-safe Base64; v6 introduces strict canonical numbers and a redaction table;
v7 adds knock; v8 adds restricted joins; v9 changes redaction preservation;
v10 adds knock-restricted joins; v11 changes redaction preservation again; and
v12 derives the room ID from its create event and uses State Resolution v2.1.

The template offers three independent validation profiles:

- `archive` retains valid, identifiable source objects and does not claim that
  they are authorized or signature-verified.
- `verified` additionally validates reference-hash IDs and signatures. It uses
  redaction rules only to construct verification inputs; it never overwrites
  raw stored events with redacted ones.
- `state-resolution` additionally evaluates authorization and creates derived
  state/redacted views.

This separation is deliberate. Canonical JSON alone is insufficient for a v3
reference hash because version-specific redaction determines its preimage.
Conversely, authorization-rule changes do not affect raw archival storage;
they matter only when a caller asks mtxdb to assert room state or event
admissibility.

## Matrix identity policy

Matrix's `event_id` is used because it is the protocol-level immutable identity
by which `prev_events`, `auth_events`, federation endpoints, and room state
refer to an event. It is more appropriate than a generic `id`, a sender, a
timestamp, or a hash of re-serialized JSON, all of which can be absent,
non-unique, mutable, or encoding-dependent. `room_id` scopes the collection so
the same storage engine can hold independent rooms.

Records without the template's identity field are not Matrix events. The Matrix
template specifies `reject_record`, rather than silently counting them as
imported or treating a wrapper object's ID as an event ID.
