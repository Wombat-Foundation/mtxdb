#!/usr/bin/env python3
"""Find the first valid Matrix `m.room.create` event in each JSONL file.

A valid create mirrors mtxdb-cli's `matrix_create_event_id()`:

  * type      == "m.room.create"
  * state_key == ""             (present and empty)
  * event_id  is a string       (present)

Each JSONL file is expected to hold one JSON event per line. The scanner
stops at the first valid create in a file, so the rest of the file is never
parsed; a cheap byte guard skips the JSON parser for lines that cannot
contain a create.

Two modes:

  find_create.py [FILE ...]        scan files directly (pure Python)
  find_create.py --candidates ...  validate `file:line:json` lines on stdin,
                                   as emitted by `rg`; FILE args are still the
                                   full file list, used to report negatives.

`find_create.sh` is the fast driver: one `rg` pass narrows the candidates to
the lines whose bytes carry `"type":"m.room.create"`, then this script
confirms the outer JSON object. Because JSON nesting is not regular, grep
alone cannot decide this; the parser is the source of truth.

Usage:

  python3 scripts/find_create.py path/to/*.jsonl
  scripts/find_create.sh path/to/*.jsonl

Exit status is 0 when every input file had a valid create, 1 otherwise.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections.abc import Iterable
from typing import Any

CANDIDATE_RE = re.compile(r"^([^:]+):(\d+):(.*)$")
MARKER = b"m.room.create"


def is_valid_create(event: Any) -> bool:
    """True when `event` is a Matrix create event with a usable event id."""
    return (
        isinstance(event, dict)
        and event.get("type") == "m.room.create"
        and event.get("state_key") == ""
        and isinstance(event.get("event_id"), str)
    )


def first_valid(candidates: Iterable[tuple[int, Any]]) -> tuple[int, dict] | None:
    """Return the lowest-numbered valid create among (line, event) pairs."""
    best: tuple[int, dict] | None = None
    for lineno, event in candidates:
        if is_valid_create(event) and (best is None or lineno < best[0]):
            best = (lineno, event)
    return best


def scan_file(path: str) -> tuple[int, dict] | None:
    """Stream `path` line by line and stop at the first valid create."""
    with open(path, "rb") as handle:
        for lineno, line in enumerate(handle, 1):
            if MARKER not in line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if is_valid_create(event):
                return lineno, event
    return None


def read_candidates(stream: Iterable[str]) -> dict[str, list[tuple[int, Any]]]:
    """Parse `file:line:json` lines (as emitted by ripgrep) into a map.

    Paths containing a colon are not supported; the corpus this targets uses
    plain names.
    """
    found: dict[str, list[tuple[int, Any]]] = {}
    for raw in stream:
        match = CANDIDATE_RE.match(raw.rstrip("\n"))
        if match is None:
            continue
        path, lineno, payload = match.group(1), int(match.group(2)), match.group(3)
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        found.setdefault(path, []).append((lineno, event))
    return found


def describe(path: str, hit: tuple[int, dict] | None) -> str:
    if hit is None:
        return f"{path}: no valid m.room.create"
    lineno, event = hit
    version = (event.get("content") or {}).get("room_version", "?")
    room = event.get("room_id", "?")
    return f"{path}:{lineno}: {event['event_id']} room={room} version={version}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Find the first valid Matrix m.room.create in each JSONL file."
    )
    parser.add_argument("paths", nargs="*", help="JSONL files (default: *.jsonl)")
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="read file:line:json candidate lines from stdin instead of scanning",
    )
    args = parser.parse_args(argv[1:])
    paths = sorted(args.paths or glob.glob("*.jsonl"))
    if not paths:
        parser.error("no input files (pass paths or run where *.jsonl exists)")

    if args.candidates:
        candidates = read_candidates(sys.stdin)
        results = [(path, first_valid(candidates.get(path, []))) for path in paths]
    else:
        results = []
        for path in paths:
            try:
                results.append((path, scan_file(path)))
            except OSError as exc:
                print(f"{path}: {exc}", file=sys.stderr)
                results.append((path, None))

    missing = sum(1 for _, hit in results if hit is None)
    for path, hit in results:
        print(describe(path, hit))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
