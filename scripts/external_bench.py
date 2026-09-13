"""Run the mtxdb / mdbx / sqlite comparison bench and print the summary table.

Appends the run to the committed `benches/csv/external.csv` history using
`compare_bench.py`'s CSV writer, so the columns, machine fingerprint, and
timestamp/`git_sha` prefix match the `make bench` history exactly. Only the
raw capture (`benches/csv/latest-external.txt`) is transient.

The comparison bench is feature-gated (`--features compare-external`), so it
is deliberately separate from the un-gated `make bench` flow.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from compare_bench import append_scenario_csv, external_scenario

ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "benches" / "csv"
LATEST = CSV_DIR / "latest-external.txt"

METRICS = [
    ("write_ms", "bulk write"),
    ("warm_open_ms", "warm open"),
    ("cold_open_ms", "cold open"),
    ("lookup_us", "point lookup"),
    ("append_ms", "batch append"),
    ("append_sync_ms", "append sync"),
    ("files_bytes", "on-disk bytes"),
]

BENCH_CMD = [
    "cargo",
    "bench",
    "--bench",
    "compare_external",
    "--features",
    "compare-external",
]


def run_bench() -> None:
    """Execute the comparison bench, capturing combined output to LATEST."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    with LATEST.open("wb") as out:
        proc = subprocess.run(
            BENCH_CMD,
            cwd=ROOT,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise SystemExit(
            f"comparison bench failed (exit {proc.returncode}); logs in {LATEST}"
        )


def append_rows() -> int:
    """Parse the run and append its external rows to benches/csv/external.csv."""
    scenario = external_scenario(LATEST.read_text(encoding="utf-8"))
    if not scenario.rows:
        print("warning: no `bench: external` rows parsed from the run", file=sys.stderr)
        return 0
    machine = (
        (CSV_DIR / "machine.txt").read_text(encoding="utf-8").strip()
        if (CSV_DIR / "machine.txt").exists()
        else "unknown"
    )
    scenario.with_machine(machine)
    append_scenario_csv(CSV_DIR, scenario)
    return len(scenario.rows)


def print_table() -> None:
    """Render the three engines side by side from the external history."""
    with (CSV_DIR / "external.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    latest: dict[str, dict] = {}
    for row in rows:
        latest[row["engine"]] = row
    engines = ["mtxdb", "mdbx", "sqlite"]
    width = max(len(m[0]) for m in METRICS)
    for engine in engines:
        print(
            f"{engine:>7}  "
            + "  ".join(f"{m[1]:>{width}}: {latest[engine][m[0]]}" for m in METRICS)
        )


def main() -> None:
    """Wire the flow: bench, append, table."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="skip the cargo bench run; only append + print from existing LATEST",
    )
    args = parser.parse_args()

    if not args.no_run:
        run_bench()
    if not LATEST.exists():
        raise SystemExit(f"{LATEST} missing; run without --no-run first")
    appended = append_rows()
    if appended:
        print_table()
    print(f"Appended {appended} rows to {CSV_DIR / 'external.csv'}.")


if __name__ == "__main__":
    main()
