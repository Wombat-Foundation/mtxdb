# mtxdb

A "write-once" `packfile` storage with optimal layout and binary format.

Running Matrix servers on spinning, mechanical drives—made _less_ painful.

- Disk offsets probed in average-case `O(1)` time via open-addressed
  linear probing.
- Binary serialization format repacked periodically in topological order, making
  nominal C2S and federation DB reads purely sequential.
- Stores edge and inverted edge indexes, facilitating near pure sequential
  segments.
