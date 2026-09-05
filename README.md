# mtxdb

A "write-once" `packfile` storage with optimal layout and binary format.

Running Matrix servers on spinning, mechanical drives—made _less_ painful.

- Exact disk offsets probed in a single `O(1)` clock cycle via open-addressed
  linear probing.
- Binary serialization format repacked periodically in topological order, making
  nominal C2S and federation DB reads purely sequential.
- Stores edge and inverted edge indexes, facilitating near pure sequential
  segments.
