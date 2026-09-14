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
    ("append_ms", "grow append"),
    ("append_sync_ms", "grow sync"),
    ("steady_append_ms", "steady append"),
    ("steady_append_sync_ms", "steady sync"),
    ("files_bytes", "on-disk bytes"),
    ("mem_bytes", "index size"),
]


def _human_bytes(value: int) -> str:
    """Compact byte count, e.g. 2099456 -> '2.1MB'."""
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{value}"


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
    columns = [m[1] for m in METRICS]

    def cell(engine: str, metric: str) -> str:
        value = latest[engine][metric]
        if metric == "files_bytes":
            return _human_bytes(int(value))
        if metric != "mem_bytes":
            # `csv` parsing normalizes `6.80` to `6.8`; restore the benchmark
            # display contract here. Bulk write intentionally remains one
            # decimal, while all other time measurements use two.
            return (
                f"{float(value):.1f}" if metric == "write_ms" else f"{float(value):.2f}"
            )
        label = latest[engine]["mem_label"]
        if label == "index_bytes":
            return f"{_human_bytes(int(value))} idx"
        # mdbx / sqlite keep their B-tree in the DB file's own mapping — the
        # "index" is not a separate resident structure.
        return "in-file"

    values = [[cell(engine, m[0]) for m in METRICS] for engine in engines]
    widths = [
        max(len(columns[i]) + 2, *(len(r[i]) for r in values))
        for i in range(len(columns))
    ]
    print(
        "engine".rjust(7)
        + "  "
        + "  ".join(label.rjust(widths[i]) for i, label in enumerate(columns))
    )
    for engine, cells in zip(engines, values):
        print(
            f"{engine:>7}  "
            + "  ".join(cell.rjust(widths[i]) for i, cell in enumerate(cells))
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


if __name__ == "__main__":
    main()
