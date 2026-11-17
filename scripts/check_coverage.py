#!/usr/bin/env python3
"""Coverage gate: a modest overall floor plus per-file floors for critical modules.

Reads the JSON from `cargo llvm-cov report --json` and fails if line coverage
of production code (test_*.rs files excluded, so tests cannot pad the number)
drops below a floor. Floors sit a few points under the measured values: they
catch a module silently losing its tests, they do not ask for tests of
defensive or unreachable lines. Raise a floor when coverage genuinely improves;
never lower one to land a change without saying why in the commit.
"""

import json
import os
import sys

OVERALL_FLOOR = 85.0

# Production paths where a regression would be a durability/recovery risk.
FILE_FLOORS = {
    "engine/journal.rs": 82.0,
    "engine/shard.rs": 80.0,
    "engine/packfile/storage.rs": 82.0,
    "engine/packfile/mod.rs": 82.0,
    "engine/database.rs": 83.0,
    "engine/index/mod.rs": 93.0,
    "engine/index/run.rs": 92.0,
    "engine/index/delta.rs": 75.0,
    "engine/index/checkpoint.rs": 88.0,
    "graph/csr/mod.rs": 95.0,
    # Auth decisions and the adjacency/closure views they read. These now
    # exercise their integrity checks (disagreement, undecodable closures,
    # false skips) and short-id round trips, not just the happy path.
    "matrix/auth_closure.rs": 84.0,
    "matrix/room_auth.rs": 86.0,
    "matrix/csr_adjacency.rs": 86.0,
}


def relative(path: str) -> str:
    marker = "/src/"
    return path.split(marker, 1)[1] if marker in path else path


def is_production(path: str) -> bool:
    name = os.path.basename(path)
    return not name.startswith("test_") and "/src/bin/" not in path


def main(report_path: str) -> int:
    with open(report_path, encoding="utf-8") as handle:
        files = json.load(handle)["data"][0]["files"]

    covered = total = 0
    per_file = {}
    for entry in files:
        path = entry["filename"]
        if not is_production(path):
            continue
        lines = entry["summary"]["lines"]
        covered += lines["covered"]
        total += lines["count"]
        per_file[relative(path)] = (lines["covered"], lines["count"])

    failures = []
    overall = 100.0 * covered / total if total else 0.0
    print(f"production line coverage: {overall:.2f}% (floor {OVERALL_FLOOR}%)")
    if overall < OVERALL_FLOOR:
        failures.append(f"overall {overall:.2f}% < {OVERALL_FLOOR}%")

    for name, floor in sorted(FILE_FLOORS.items()):
        if name not in per_file:
            failures.append(f"{name}: missing from the coverage report")
            continue
        hit, count = per_file[name]
        pct = 100.0 * hit / count if count else 100.0
        status = "ok " if pct >= floor else "LOW"
        print(f"  {status} {name}: {pct:.2f}% (floor {floor}%)")
        if pct < floor:
            failures.append(f"{name} {pct:.2f}% < {floor}%")

    if failures:
        print("coverage gate failed:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ".coverage/summary.json"))
