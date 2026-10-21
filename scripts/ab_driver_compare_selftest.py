#!/usr/bin/env python3
"""Offline selftest for scripts/ab_driver_compare.py.

Feeds the parser the exact block shape the driver emits, asserts the
parsed snapshot, then runs the A/B table render so the delta columns
line up.  No HDD or Synapse workload needed.
"""

from ab_driver_compare import fmt_bytes, parse

# Real captured shape from a tests.handlers run.
LEFT = """\
=== FFI timing ===
  ffi_event_json_put     266.644ms  2360  0.099ms  0.113ms  0.217ms
  ffi_event_edges_put    203.088ms  1842  0.099ms  0.100ms  0.177ms
  ffi_put_hamt_nodes     321.411ms  5698  0.114ms  0.056ms  0.180ms
  ffi_event_json_get     145.902ms  8196  0.018ms  0.012ms  0.041ms

=== mtxdb runtime stats ===
  [event_json]
    collections: 561  shards: 1  index: 1,225,232B
    put: 0 calls, 0B | put_many: 2901 calls, 3036 records, 479,900B
    sync: 12 calls | sync_all: 10113 calls, 4490 direct sidecar writes
  [edge_dag]
    collections: 3659  shards: 256
    put_many: 10439 calls, 142384 records, 22952338B
    sync: 1170 calls | sync_all: 9054 calls, 4490 direct sidecar writes
"""

RIGHT = """\
=== FFI timing ===
  ffi_event_json_put     310.200ms  2360  0.131ms  0.113ms  0.217ms
  ffi_event_edges_put    250.000ms  1842  0.136ms  0.100ms  0.177ms
  ffi_put_hamt_nodes     400.000ms  5698  0.140ms  0.056ms  0.180ms
  ffi_event_json_get     150.000ms  8196  0.018ms  0.012ms  0.041ms

=== mtxdb runtime stats ===
  [event_json]
    collections: 561  shards: 1  index: 1,225,232B
    put: 0 calls, 0B | put_many: 3100 calls, 3300 records, 520000B
    sync: 18 calls | sync_all: 12000 calls, 5000 direct sidecar writes
  [edge_dag]
    collections: 3659  shards: 256
    put_many: 11000 calls, 150000 records, 24000000B
    sync: 1300 calls | sync_all: 10000 calls, 5000 direct sidecar writes
"""


def test_parse_left() -> None:
    ds = parse(LEFT)
    assert ds.put_many_calls == 2901 + 10439, f"put_many_calls={ds.put_many_calls}"
    assert ds.put_many_records == 3036 + 142384, (
        f"put_many_records={ds.put_many_records}"
    )
    assert ds.sync_calls == 12 + 1170, f"sync_calls={ds.sync_calls}"
    assert ds.sync_all_calls == 10113 + 9054, f"sync_all_calls={ds.sync_all_calls}"
    assert "ffi_event_json_put" in ds.ffi_times
    assert ds.ffi_times["ffi_event_json_put"] == 266.644
    assert ds.ffi_calls["ffi_event_json_put"] == 2360
    assert ds.ffi_calls["ffi_event_json_get"] == 8196
    print("  test_parse_left: PASSED")


def test_parse_right() -> None:
    ds = parse(RIGHT)
    assert ds.put_many_calls == 3100 + 11000, f"put_many_calls={ds.put_many_calls}"
    assert ds.put_many_records == 3300 + 150000, (
        f"put_many_records={ds.put_many_records}"
    )
    assert ds.sync_calls == 18 + 1300, f"sync_calls={ds.sync_calls}"
    assert ds.sync_all_calls == 12000 + 10000, f"sync_all_calls={ds.sync_all_calls}"
    print("  test_parse_right: PASSED")


def test_delta_direction() -> None:
    left = parse(LEFT)
    right = parse(RIGHT)
    assert right.put_many_calls > left.put_many_calls
    assert right.put_many_records > left.put_many_records
    assert right.sync_calls > left.sync_calls
    assert right.sync_all_calls > left.sync_all_calls
    print("  test_delta_direction: PASSED")


def test_fmt_bytes() -> None:
    assert fmt_bytes(0) == "0B"
    assert fmt_bytes(1023) == "1023B"
    assert fmt_bytes(1024) == "1KiB"
    assert fmt_bytes(1048576) == "1MiB"
    print("  test_fmt_bytes: PASSED")


def test_merge() -> None:
    a = parse(LEFT)
    b = parse(RIGHT)
    a.merge(b)
    assert a.put_many_calls == parse(LEFT).put_many_calls + parse(RIGHT).put_many_calls
    assert a.sync_calls == parse(LEFT).sync_calls + parse(RIGHT).sync_calls
    print("  test_merge: PASSED")


if __name__ == "__main__":
    test_parse_left()
    test_parse_right()
    test_delta_direction()
    test_fmt_bytes()
    test_merge()
    print("\nAll selftests passed.")
