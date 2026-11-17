#!/usr/bin/env python3
"""Measure D (split) and A from an `mtxdb-export/v1` room.

`A` is the total number of auth-ancestor memberships materialized across every
event's transitive auth closure: `sum(|ancestors(event)|)`. It is computed from
the export's `auth_events` graph, without running the importer, so it can be
compared between a real room and a generated fixture.

A referenced id that is absent from the export counts as one ancestor (the walk
cannot continue past it), matching the importer's inability to traverse it.

Usage:
  python3 scripts/measure_export_closures.py room-*.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(path: Path) -> tuple[dict[str, list[str]], dict[str, int]]:
    auth: dict[str, list[str]] = {}
    prev_degree: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("t") != "event":
            continue
        event_id = record["event_id"]
        auth[event_id] = list(record.get("auth_events", []))
        prev_degree[event_id] = len(record.get("prev_events", []))
    return auth, prev_degree


def closures(auth: dict[str, list[str]]) -> dict[str, frozenset[str]]:
    memo: dict[str, frozenset[str]] = {}

    def visit(event_id: str, stack: frozenset[str]) -> frozenset[str]:
        cached = memo.get(event_id)
        if cached is not None:
            return cached
        if event_id in stack:
            # A cycle is not expected in an export; stop rather than loop.
            return frozenset()
        result: set[str] = set()
        for parent in auth.get(event_id, ()):
            result.add(parent)
            result.update(visit(parent, stack | {event_id}))
        frozen = frozenset(result)
        memo[event_id] = frozen
        return frozen

    return {event_id: visit(event_id, frozenset()) for event_id in auth}


def quantiles(values: list[int], points: tuple[float, ...]) -> dict[str, int]:
    if not values:
        return {f"p{int(p * 100)}": 0 for p in points}
    ordered = sorted(values)
    return {
        f"p{int(p * 100)}": ordered[
            min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
        ]
        for p in points
    }


def analyse(path: Path) -> dict:
    auth, prev_degree = load(path)
    event_closures = closures(auth)
    sizes = [len(value) for value in event_closures.values()]
    events = len(auth)
    return {
        "file": str(path),
        "events": events,
        "d_prev": sum(prev_degree.values()),
        "d_auth": sum(len(value) for value in auth.values()),
        "a": sum(sizes),
        "a_per_event": round(sum(sizes) / events, 2) if events else 0.0,
        "max_closure": max(sizes) if sizes else 0,
        "closure_quantiles": quantiles(sizes, (0.5, 0.9, 0.99)),
        "closure_mean": round(statistics.fmean(sizes), 2) if sizes else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rooms", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [analyse(path) for path in args.rooms]
    text = json.dumps(rows, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
