# Changelog

## Unreleased

### Added: content-addressed HAMTs

mtxdb bundles `rezzy`'s persistent HAMT (CHAMP) for content-addressed trees:

- `mtxdb::hamt` re-exports the trie. It is generic over any key and value
  types with a `HamtCodec` encoding, and routes keys by their encoded bytes, so
  equal entries give byte-identical nodes everywhere.
- `mtxdb::hamt_store` persists those trees in a collection: each node is a
  record under the first 16 bytes of its structural hash, verified against the
  full hash on every read, and `put_nodes` stages a tree into a
  `DatabaseTransaction` so it commits with the roots that name it.
- `mtxdb::state_hamt` is the Matrix room-state layer on top: `MTHR` root
  records and `MTHN` nodes in a room's `STAT` collection, shared by the
  importer and server integrations.

### Platform support

- The HAMT modules build on every target, including 32-bit x86: `rezzy` and
  `rezzy-recon` are regular dependencies with their default features there too.
