#!/usr/bin/env python3
"""Compare verified persisted auth children with an ephemeral reverse map.

Synthetic data only. Executes the three relevant Sithnapse Python functions via
AST extraction, without importing or changing a homeserver. SQLite stands in for
both the embedded point store and SQL edge table. The alternative is a Python
implementation of Congruent's traversal algorithm, not its Rust implementation.
Reported reads are logical requests; bytes are returned payloads, not disk pages.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import json
import sqlite3
import statistics
import struct
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Reads:
    forward_reads: int = 0
    embedded_reads: int = 0
    identity_probes: int = 0
    sql_queries: int = 0
    payload_bytes: int = 0
    child_ids_scanned: int = 0
    repaired_rows: int = 0


def pack(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}I", *values)


def unpack(value: bytes) -> list[int]:
    return list(struct.unpack(f"<{len(value) // 4}I", value))


class Fixture:
    def __init__(self, size: int, outside_factor: int, incomplete_percent: int):
        self.size = size
        total = size * (outside_factor + 1)
        self.ids = [
            "$"
            + base64.urlsafe_b64encode(hashlib.sha256(str(i).encode()).digest())
            .decode()
            .rstrip("=")
            for i in range(total)
        ]
        self.shorts = {event: short for short, event in enumerate(self.ids)}
        self.parents: list[list[int]] = []
        children: list[list[int]] = [[] for _ in range(total)]
        for child in range(total):
            if child == 0:
                parents = []
            elif child < size:
                # State-event auth chain with piecewise and interleaved hubs.
                # No create edge. This is DAG-shaped, not protocol-authorized PDUs.
                parents = sorted(
                    {child - 1, ((child - 1) // 256) * 256, max(0, child - 64)}
                )
            else:
                # Children outside the ancestor union test candidate pruning.
                parents = [child % size]
            self.parents.append(parents)
            for parent in parents:
                children[parent].append(child)
        self.initial_children = [
            pack([child for child in row if child % 100 >= incomplete_percent])
            for row in children
        ]
        self.db = sqlite3.connect(":memory:")
        self.db.executescript("""
            CREATE TABLE records (
                short INTEGER PRIMARY KEY, parents BLOB NOT NULL, children BLOB NOT NULL
            );
            CREATE TABLE event_auth (event_id TEXT NOT NULL, auth_id TEXT NOT NULL);
            CREATE INDEX event_auth_parent ON event_auth(auth_id);
        """)
        self.db.executemany(
            "INSERT INTO records VALUES (?, ?, ?)",
            (
                (i, pack(row), self.initial_children[i])
                for i, row in enumerate(self.parents)
            ),
        )
        self.db.executemany(
            "INSERT INTO event_auth VALUES (?, ?)",
            (
                (self.ids[child], self.ids[parent])
                for child, row in enumerate(self.parents)
                for parent in row
            ),
        )
        self.db.commit()
        self.reads = Reads()
        self.modified: set[int] = set()
        self.namespace: dict[str, Any] = {}

    def reset(self) -> None:
        self.db.executemany(
            "UPDATE records SET children = ? WHERE short = ?",
            ((self.initial_children[parent], parent) for parent in self.modified),
        )
        self.db.commit()
        self.modified.clear()
        self.reads = Reads()

    def forward(self, short: int) -> list[int]:
        self.reads.forward_reads += 1
        payload = self.db.execute(
            "SELECT parents FROM records WHERE short = ?", (short,)
        ).fetchone()[0]
        self.reads.payload_bytes += len(payload)
        return unpack(payload)

    def embedded_children(
        self, _engine: Any, _namespace: str, _room: str, parents: list[int]
    ) -> dict[int, list[int]]:
        result = {}
        for parent in parents:
            self.reads.embedded_reads += 1
            payload = self.db.execute(
                "SELECT children FROM records WHERE short = ?", (parent,)
            ).fetchone()[0]
            self.reads.payload_bytes += len(payload)
            result[parent] = unpack(payload)
        return result

    def resolve_ids(
        self, _engine: Any, _namespace: str, _room: str, shorts: list[int]
    ) -> list[str]:
        self.reads.identity_probes += len(shorts)
        events = [self.ids[short] for short in shorts]
        self.reads.payload_bytes += sum(len(event.encode()) for event in events)
        return events

    def get_shorts(
        self, _engine: Any, _namespace: str, _room: str, events: list[str]
    ) -> list[int]:
        self.reads.identity_probes += len(events)
        self.reads.payload_bytes += 4 * len(events)
        return [self.shorts[event] for event in events]

    def auth_chain_children_append(
        self, _namespace: str, _room: str, rows: list[tuple[int, list[int]]]
    ) -> None:
        for parent, additions in rows:
            payload = self.db.execute(
                "SELECT children FROM records WHERE short = ?", (parent,)
            ).fetchone()[0]
            self.reads.embedded_reads += 1
            self.reads.payload_bytes += len(payload)
            self.db.execute(
                "UPDATE records SET children = ? WHERE short = ?",
                (pack(sorted(set(unpack(payload)) | set(additions))), parent),
            )
            self.modified.add(parent)
            self.reads.repaired_rows += 1

    # The original SQL helper expects a transaction with execute/fetchall.
    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self.reads.sql_queries += 1
        self.cursor = self.db.execute(sql, params)

    def fetchall(self) -> list[tuple[str]]:
        rows = self.cursor.fetchall()
        self.reads.payload_bytes += sum(len(event.encode()) for (event,) in rows)
        self.reads.child_ids_scanned += len(rows)
        return rows


def bind_sithnapse(fixture: Fixture, source: Path) -> str:
    text = source.read_text()
    names = {
        "_fetch_child_event_ids_from_sql",
        "_fetch_and_verify_children",
        "get_forward_reachable_short_ids",
    }
    tree = ast.parse(text, filename=str(source))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in definitions} != names:
        raise RuntimeError(
            "Sithnapse helper names changed; re-audit the benchmark adapter"
        )
    namespace: dict[str, Any] = {
        "get_embedded_auth_children_batch": fixture.embedded_children,
        "resolve_short_ids_to_event_ids": fixture.resolve_ids,
        "get_or_create_short_ids": fixture.get_shorts,
        "_engine": lambda _name: fixture,
    }
    # Postpone annotations so extraction does not import homeserver dependencies.
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *definitions,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    fixture.namespace = namespace
    return hashlib.sha256(text.encode()).hexdigest()


def ancestors(
    fixture: Fixture, seeds: list[int], build_reverse: bool
) -> tuple[set[int], dict[int, list[int]]]:
    seen = set(seeds)
    todo = list(seeds)
    reverse: dict[int, list[int]] = {}
    while todo:
        child = todo.pop()
        for parent in fixture.forward(child):
            if build_reverse:
                reverse.setdefault(parent, []).append(child)
            if parent not in seen:
                seen.add(parent)
                todo.append(parent)
    return seen, reverse


def ephemeral(
    fixture: Fixture, seeds: list[int], cached_candidates: set[int] | None
) -> set[int]:
    if cached_candidates is None:
        candidates, reverse = ancestors(fixture, seeds, True)
    else:
        candidates = cached_candidates
        reverse = {}
        for child in sorted(candidates):
            for parent in fixture.forward(child):
                reverse.setdefault(parent, []).append(child)
    reached: set[int] = set()
    todo = list(seeds)
    while todo:
        parent = todo.pop()
        if parent in reached:
            continue
        reached.add(parent)
        todo.extend(reverse.get(parent, ()))
    return reached & candidates


def persisted(
    fixture: Fixture, seeds: list[int], cached_candidates: set[int] | None
) -> set[int]:
    if cached_candidates is None:
        candidates, _ = ancestors(fixture, seeds, False)
    else:
        candidates = cached_candidates
    return (
        fixture.namespace["get_forward_reachable_short_ids"](
            fixture, None, "bench", "!synthetic", seeds, candidates
        )
        & candidates
    )


def measure(
    fixture: Fixture, seeds: list[int], mode: str, strategy: str, repeats: int
) -> dict[str, Any]:
    cached = set(range(fixture.size)) if mode == "cached_candidates" else None
    expected = set(range(min(seeds), fixture.size))  # independent chain oracle
    operation = persisted if strategy == "persisted_verified" else ephemeral
    times = []
    metrics = None
    for _ in range(repeats):
        fixture.reset()
        start = time.perf_counter_ns()
        actual = operation(fixture, seeds, cached)
        times.append((time.perf_counter_ns() - start) / 1e6)
        if actual != expected:
            raise AssertionError(f"incorrect subgraph: {mode}, {strategy}")
        current = asdict(fixture.reads)
        if metrics is not None and metrics != current:
            raise AssertionError("read counts changed across reset samples")
        metrics = current
    fixture.reset()
    tracemalloc.start()
    actual = operation(fixture, seeds, cached)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if actual != expected:
        raise AssertionError("incorrect subgraph in memory sample")
    assert metrics is not None
    return {
        "median_ms": round(statistics.median(times), 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "python_peak_bytes": peak,
        "result_nodes": len(actual),
        **metrics,
    }


def verify_contract(source: Path) -> None:
    """Exercise forks, candidate pruning, seed inclusion and SQL repair."""
    fixture = Fixture(8, 0, 100)
    fixture.parents = [[], [0], [0], [1, 2], [3], [2], [4, 5], [1]]
    fixture.initial_children = [b""] * 8
    fixture.db.execute("DELETE FROM records")
    fixture.db.execute("DELETE FROM event_auth")
    fixture.db.executemany(
        "INSERT INTO records VALUES (?, ?, ?)",
        ((i, pack(row), b"") for i, row in enumerate(fixture.parents)),
    )
    fixture.db.executemany(
        "INSERT INTO event_auth VALUES (?, ?)",
        (
            (fixture.ids[child], fixture.ids[parent])
            for child, row in enumerate(fixture.parents)
            for parent in row
        ),
    )
    fixture.db.commit()
    bind_sithnapse(fixture, source)
    expected = {1, 3, 4, 6}
    for candidates in [None, set(range(7))]:
        fixture.reset()
        actual = persisted(fixture, [1, 6], candidates)
        if actual != expected or fixture.reads.repaired_rows == 0:
            raise AssertionError("persisted fork/pruning/repair contract failed")
        fixture.reset()
        if ephemeral(fixture, [1, 6], candidates) != expected:
            raise AssertionError("ephemeral fork/pruning contract failed")
    fixture.db.close()


def integers(value: str) -> list[int]:
    return [int(part) for part in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sithnapse-source",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "sithnapse/synapse/storage/databases/main/embedded_event_auth_chains.py",
    )
    parser.add_argument("--sizes", type=integers, default=[1000, 10000, 100000])
    parser.add_argument("--conflicts", type=integers, default=[2, 8, 32])
    parser.add_argument(
        "--seed-patterns",
        type=lambda value: value.split(","),
        default=["spanning", "tips"],
    )
    parser.add_argument("--outside-factor", type=int, default=1)
    parser.add_argument("--incomplete-percent", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not set(args.seed_patterns) <= {"spanning", "tips"}:
        parser.error("seed patterns must be spanning or tips")
    if (
        min(args.sizes) < 4
        or min(args.conflicts) < 2
        or max(args.conflicts) > min(args.sizes) // 2
        or args.outside_factor < 0
        or not 0 <= args.incomplete_percent <= 100
        or args.repeats < 1
    ):
        parser.error("invalid fixture dimensions")
    verify_contract(args.sithnapse_source)
    results = []
    for size in args.sizes:
        fixture = Fixture(size, args.outside_factor, args.incomplete_percent)
        source_sha = bind_sithnapse(fixture, args.sithnapse_source)
        for conflicts in args.conflicts:
            for pattern in args.seed_patterns:
                seeds = (
                    [
                        size // 2 + (size - 1 - size // 2) * i // (conflicts - 1)
                        for i in range(conflicts)
                    ]
                    if pattern == "spanning"
                    else list(range(size - conflicts, size))
                )
                for mode in ["cached_candidates", "discover_candidates"]:
                    for strategy in ["persisted_verified", "ephemeral_forward"]:
                        row = {
                            "union_nodes": size,
                            "room_nodes": len(fixture.parents),
                            "conflicts": conflicts,
                            "seed_pattern": pattern,
                            "incomplete_percent": args.incomplete_percent,
                            "mode": mode,
                            "strategy": strategy,
                            "repeats": args.repeats,
                            **measure(fixture, seeds, mode, strategy, args.repeats),
                        }
                        results.append(row)
                        print(json.dumps(row), flush=True)
        fixture.db.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(results[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(results)
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "sithnapse_source_sha256": source_sha,
                "sizes": args.sizes,
                "conflicts": args.conflicts,
                "outside_factor": args.outside_factor,
                "seed_patterns": args.seed_patterns,
                "incomplete_percent": args.incomplete_percent,
                "python": __import__("sys").version,
                "sqlite": sqlite3.sqlite_version,
                "scope": "Python helper/algorithm comparison with synthetic SQLite "
                "adapter; no homeserver DB, mtxdb or Rust timing, no physical IO "
                "measurement",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
