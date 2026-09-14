"""Run the mtxdb / mdbx / sqlite comparison bench and print the summary table.

Running the script directly only prints the table; it never touches the
committed `benches/csv/external.csv` history. Pass `--append` (as
`make _bench/external` does) to append the run using `compare_bench.py`'s
CSV writer, so the columns, machine fingerprint, and timestamp/`git_sha`
prefix match the `make bench` history exactly. Only the raw capture
(`benches/csv/latest-external.txt`) is transient.

The comparison bench is feature-gated (`--features compare-external`), so it
is deliberately separate from the un-gated `make bench` flow.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from compare_bench import Scenario, append_scenario_csv, external_scenario

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
    # PSS (proportional set size, /proc/self/smaps_rollup) is the one memory
    # number captured identically for all three engines, so it is the
    # cross-engine-comparable column; "index size" above is not (see its
    # mem_label: resident index bytes for mtxdb, on-disk file bytes for
    # mdbx/sqlite). "mem open" is sampled right after the warm open, before
    # any lookups touch pages; "mem warm" after the sampled lookup pass.
    ("pss_open_bytes", "mem open"),
    ("pss_warm_bytes", "mem warm"),
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

ENGINES = ("mtxdb", "mdbx", "sqlite")


def run_bench() -> None:
    """Run the comparison bench once per engine, each its own process.

    `/proc/self/smaps_rollup`-based RSS/PSS sampling (see
    `compare_external.rs::smaps_rollup`) reports the whole process, so
    running all three engines in one `cargo bench` invocation would let
    allocator retention and a prior engine's still-resident pages bias
    later engines' numbers. `MTXDB_BENCH_EXT_ENGINE` restricts a run to a
    single backend; running it three times, once per engine, gives each
    one a fresh address space for that measurement. Output from all three
    processes is concatenated into LATEST so `append_rows` parses it the
    same as a single combined run.
    """
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    with LATEST.open("wb") as out:
        for engine in ENGINES:
            env = {**os.environ, "MTXDB_BENCH_EXT_ENGINE": engine}
            proc = subprocess.run(
                BENCH_CMD,
                cwd=ROOT,
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
            if proc.returncode != 0:
                raise SystemExit(
                    f"comparison bench failed for engine={engine} "
                    f"(exit {proc.returncode}); logs in {LATEST}"
                )


def append_rows(scenario: Scenario) -> int:
    """Tag a parsed run with its machine fingerprint and append it to
    `benches/csv/external.csv`."""
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


def print_table(rows: list[dict]) -> None:
    """Render the three engines side by side from a run's external rows."""
    latest: dict[str, dict] = {}
    for row in rows:
        latest[row["engine"]] = row
    engines = ["mtxdb", "mdbx", "sqlite"]
    columns = [m[1] for m in METRICS]

    # Keep the terminal summary compact now that it shows both the structural
    # grow cost and the ordinary steady-state append cost. Metric labels are
    # intentionally two words at most, so render them as a two-line header
    # without widening a column for the combined phrase.
    header_rows = [
        tuple(label.split(maxsplit=1)) if " " in label else ("", label)
        for label in columns
    ]

    def cell(engine: str, metric: str) -> str:
        value = latest[engine][metric]
        if metric in ("files_bytes", "pss_open_bytes", "pss_warm_bytes"):
            return _human_bytes(int(value))
        if metric != "mem_bytes":
            # `csv` parsing normalizes `6.800` to `6.8`; restore the benchmark
            # display contract here. Bulk write intentionally remains one
            # decimal, warm/cold open use three (their spreads are small),
            # and all other time measurements use two.
            if metric == "write_ms":
                return f"{float(value):.1f}"
            if metric in ("warm_open_ms", "cold_open_ms"):
                return f"{float(value):.3f}"
            return f"{float(value):.2f}"
        label = latest[engine]["mem_label"]
        if label == "index_bytes":
            return f"{_human_bytes(int(value))} idx"
        # mdbx / sqlite keep their B-tree in the DB file's own mapping — the
        # "index" is not a separate resident structure.
        return "in-file"

    values = [[cell(engine, m[0]) for m in METRICS] for engine in engines]
    widths = [
        max(
            *(len(word) for word in header_rows[i]),
            *(len(row[i]) for row in values),
        )
        for i in range(len(columns))
    ]

    print()
    print(
        "".rjust(7)
        + "  "
        + "  ".join(header_rows[i][0].rjust(widths[i]) for i in range(len(columns)))
    )
    print(
        "engine".rjust(7)
        + "  "
        + "  ".join(header_rows[i][1].rjust(widths[i]) for i in range(len(columns)))
    )
    print()
    for engine, cells in zip(engines, values):
        print(
            f"{engine:>7}  "
            + "  ".join(cell.rjust(widths[i]) for i, cell in enumerate(cells))
        )


def main() -> None:
    """Wire the flow: bench, then (optionally) append history, then table."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="skip the cargo bench run; only append + print from existing LATEST",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="append the run to benches/csv/external.csv history (default: skip)",
    )
    args = parser.parse_args()

    if not args.no_run:
        run_bench()
    if not LATEST.exists():
        raise SystemExit(f"{LATEST} missing; run without --no-run first")
    scenario = external_scenario(LATEST.read_text(encoding="utf-8"))
    if not scenario.rows:
        print("warning: no `bench: external` rows parsed from the run", file=sys.stderr)
        return
    if args.append:
        append_rows(scenario)
    print_table(scenario.rows)


if __name__ == "__main__":
    main()
