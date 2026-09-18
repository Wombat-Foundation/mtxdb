#!/usr/bin/env python3
"""A/B-compare two mtxdb driver runs using ONLY the blocks the driver already
prints at the end of every `tests.handlers` suite run.

This reads nothing from the disk, mounts nothing, and toggles no policy — it
is a pure post-hoc parser so the A/B can be coordinated on paper/in-CI before
the HDD is in the picture.

Blocks parsed (each is a real driver emission):

  === FFI batch counters ===   (per-namespace FFI timing rows)
    event_json_put_many  275.8ms  1842  1842  0.150ms
    format: name total_ms calls records avg_ms

  === mtxdb runtime stats ===  (per-collection put/sync batches)
    [event_json]
      put: 0 calls, 0B | put_many: 12,411 calls, 23,739 records, 34,832,984B
      sync: 12 calls | sync_all: 19,167 calls, 1330 direct sidecar writes

  === State store mtxdb-vs-SQL timings ===  (optional; only when the
      embedded HAMT engine is enabled for state)

  The comparison focuses on the write-durability drumbeat the transcript
  identified as the HDD-noise driver:

    - put_many batch volume (records + bytes)   -> eager page-cache churn
    - sync  calls                                -> explicit per-sync barriers
    - sync_all calls                             -> full-durability barriers
    - shard writes + bytes                       -> positioned pack writes

  Usage (after two runs, `--left` = run A, `--right` = run B):

    python3 scripts/ab_driver_compare.py --left runA.txt --right runB.txt

  Each side must have previously run the identical Synapse commit + workload
  (the same `tests.handlers` suite) with mtxdb attached in BOTH runs; a
  control where mtxdb is disabled is NOT equivalent (it drops partial-state
  tests) and must not be mixed into this comparison.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

# A real FFI timing row, e.g.:
#   event_json_put_many  275.8ms  1842  1842  0.150ms
#   (optional avg + p50 columns are ignored)
TIMING_ROW = re.compile(
    r"^\s*(?P<name>ffi_[a-z0-9_]+|[a-z][a-z0-9_]*)\s+"
    r"(?P<ms>[0-9]+(?:\.[0-9]+)?)ms\s+"
    r"(?P<calls>[0-9][0-9,]*)",
    re.MULTILINE,
)

# put / put_many line inside a collection block:
#   put: 0 calls, 0B | put_many: 12,411 calls, 23,739 records, 34,832,984B
#   put_many: 10439 calls, 142384 records, 22952338B          (standalone)
BATCH_LINE = re.compile(
    r"(?:\|\s*)?put_many:\s+(?P<calls>[0-9][0-9,]*)\s+calls,\s+"
    r"(?P<records>[0-9][0-9,]*)\s+records,\s+"
    r"(?P<bytes>[0-9][0-9,]*)B",
    re.MULTILINE,
)

# sync / sync_all line inside a collection block:
#   sync: 12 calls | sync_all: 19,167 calls, 1330 direct sidecar writes
SYNC_LINE = re.compile(
    r"sync:\s+(?P<sync>[0-9][0-9,]*)\s+calls\s*\|\s+"
    r"sync_all:\s+(?P<sync_all>[0-9][0-9,]*)\s+calls",
    re.MULTILINE,
)

# Generic put: N calls line (FFI batch counters "put: N calls").
PUT_LINE = re.compile(
    r"put:\s+(?P<calls>[0-9][0-9,]*)\s+calls",
    re.MULTILINE,
)


def _num(text: str) -> int:
    return int(text.replace(",", ""))


@dataclass
class FfiRow:
    name: str
    total_ms: float
    calls: int


@dataclass
class Dataset:
    """Aggregates the write-durability drumbeat across a driver run."""

    ffi_times: dict[str, float] = field(default_factory=dict)
    ffi_calls: dict[str, int] = field(default_factory=dict)
    # per-collection
    cycalls: dict[str, int] = field(default_factory=dict)
    cyz: dict[str, int] = field(default_factory=dict)
    sync_calls: int = 0
    sync_all_calls: int = 0
    # rolling tally of put_many totals (absent per-collection header -> still
    # parseable from the FFI timing rows)
    put_many_calls: int = 0
    put_many_records: int = 0
    put_many_bytes: int = 0
    # top-level "put: N calls" line, when present (event_json block)
    put_calls: int = 0

    def merge(self, other: "Dataset") -> None:
        for k, v in other.ffi_times.items():
            self.ffi_times[k] = self.ffi_times.get(k, 0.0) + v
        for k, v in other.ffi_calls.items():
            self.ffi_calls[k] = self.ffi_calls.get(k, 0) + v
        for k, v in other.cycalls.items():
            self.cycalls[k] = self.cycalls.get(k, 0) + v
        for k, v in other.cyz.items():
            self.cyz[k] = self.cyz.get(k, 0) + v
        self.sync_calls += other.sync_calls
        self.sync_all_calls += other.sync_all_calls
        self.put_many_calls += other.put_many_calls
        self.put_many_records += other.put_many_records
        self.put_many_bytes += other.put_many_bytes
        self.put_calls += other.put_calls


# ---------------------------------------------------------------------------
# Table of the metrics that matter, computed WITHOUT any disk access.
# ---------------------------------------------------------------------------

_IMPROVE_LOWER = {
    # smaller is better for everything we track here (durability drumbeat)
}


def _delta_text(a: float, b: float) -> str:
    d = b - a
    return f"{d:+.0f}" if abs(d) >= 0.5 else f"{d:+.3f}"


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n = n // 1024
    return f"{n:.0f}MiB"


def snapshot(name: str, ds: Dataset) -> None:
    print(f"[{name}]")
    print(
        f"  put_many: {ds.put_many_calls} calls, {ds.put_many_records} records, "
        f"{fmt_bytes(ds.put_many_bytes)}"
    )
    print(f"  sync: {ds.sync_calls} calls | sync_all: {ds.sync_all_calls} calls")
    print(f"  put (direct): {ds.put_calls} calls")
    if ds.cycalls:
        print("  per-collection put_many calls:")
        for col, calls in sorted(ds.cycalls.items()):
            record = ds.cyz.get(col, 0)
            print(f"    {col:<28}{calls:>8} calls, {record:>8} records")


def parse(text: str) -> Dataset:
    ds = Dataset()
    for m in TIMING_ROW.finditer(text):
        name, ms, calls = (
            m.group("name"),
            float(m.group("ms")),
            _num(m.group("calls")),
        )
        ds.ffi_times[name] = ds.ffi_times.get(name, 0.0) + ms
        ds.ffi_calls[name] = ds.ffi_calls.get(name, 0) + calls
        if name in (
            "event_json_put_many",
            "event_edges_put_many",
            "ffi_event_json_put_many",
            "ffi_event_edges_put_many",
        ):
            ds.put_many_calls += calls
    for m in BATCH_LINE.finditer(text):
        ds.put_many_calls += _num(m.group("calls"))
        ds.put_many_records += _num(m.group("records"))
        ds.put_many_bytes += _num(m.group("bytes"))
    for m in SYNC_LINE.finditer(text):
        ds.sync_calls += _num(m.group("sync"))
        ds.sync_all_calls += _num(m.group("sync_all"))
    for m in PUT_LINE.finditer(text):
        ds.put_calls += _num(m.group("calls"))
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True, type=Path)
    ap.add_argument("--right", required=True, type=Path)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    left = parse(args.left.read_text(errors="replace"))
    right = parse(args.right.read_text(errors="replace"))

    # The transcript's dominant signal: per-sync barrier volume.
    # Compute relative change with a floor so a zero-baseline doesn't div-zero.
    def _rel(a: int, b: int) -> str:
        if a == 0 and b == 0:
            return "n/a"
        denom = a or 1
        return f"{(b - a) / denom:+.1%}"

    print("=== mtxdb durability drumbeat (A/B) ===")
    print(f"{'metric':<28}{'left':>16}{'right':>16}{'delta':>12}")
    rows = [
        ("put_many calls", left.put_many_calls, right.put_many_calls),
        ("put_many records", left.put_many_records, right.put_many_records),
        ("put_many bytes", left.put_many_bytes, right.put_many_bytes),
        ("sync calls", left.sync_calls, right.sync_calls),
        ("sync_all calls", left.sync_all_calls, right.sync_all_calls),
    ]
    for label, left_value, right_value in rows:
        print(
            f"{label:<28}{left_value:>16,}{right_value:>16,}"
            f"{_rel(left_value, right_value):>12}"
        )
    if args.verbose:
        print()
        snapshot("left", left)
        print()
        snapshot("right", right)

    if args.csv:
        with args.csv.open("w") as f:
            f.write("metric,left,right,delta\n")
            for label, left_value, right_value in rows:
                f.write(
                    f"{label},{left_value},{right_value},"
                    f"{_rel(left_value, right_value)}\n"
                )

    # Heuristic regression gate: if every durability signal grew, flag it.
    grew = all(
        (
            right.put_many_calls > left.put_many_calls,
            right.put_many_records > left.put_many_records,
            right.sync_calls > left.sync_calls,
            right.sync_all_calls > left.sync_all_calls,
        )
    )
    if grew:
        print(
            "\nNOTE: all four durability metrics increased; this is the "
            "eager-per-write signature worth isolating. Check whether the "
            "right run repeated a workload that the left run single-passed "
            "(e.g. partial-state retries double-persist)."
        )


if __name__ == "__main__":
    main()
