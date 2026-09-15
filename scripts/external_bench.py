"""Run the mtxdb / mdbx / sqlite / fjall comparison bench and print the summary table.

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
    ("write_ms", "bulk write (ms)"),
    ("warm_open_ms", "warm open (ms)"),
    ("cold_open_ms", "cold checkpoint (ms)"),
    ("lookup_us", "point lookup (μs)"),
    ("append_ms", "first append (ms)"),
    ("append_sync_ms", "first sync (ms)"),
    ("steady_append_ms", "steady append (ms)"),
    ("steady_append_sync_ms", "steady sync (ms)"),
    ("files_bytes", "on-disk bytes ~"),
    ("mem_bytes", "index size ~"),
    # PSS (proportional set size, /proc/self/smaps_rollup) is the one memory
    # number captured identically for all three engines, so it is the
    # cross-engine-comparable column; "index size" above is not (see its
    # mem_label: resident index bytes for mtxdb, on-disk file bytes for
    # mdbx/sqlite/fjall). "mem open" is sampled right after the warm open, before
    # any lookups touch pages; "mem warm" after the sampled lookup pass.
    ("pss_open_bytes", "mem open ~"),
    ("pss_warm_bytes", "mem warm ~"),
    # The frame-level ChecksumPolicy actually in effect: "full"/"writeonly"/
    # "none" for mtxdb (matching its row -- see INVOCATIONS), "na" for
    # mdbx/sqlite/fjall, which have no equivalent read-time integrity check.
    # Deliberately the frame policy, not the checkpoint's own (checkpoints
    # have no fully-off tier, so that would read "writeonly" for both the
    # "no crc" and "writeonly" rows and the column couldn't tell them apart).
    ("checksum", "crc check mode"),
]


def _human_bytes(value: int) -> str:
    """Compact byte count, e.g. 2099456 -> '2.1MB'."""
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024


BENCH_CMD = [
    "cargo",
    "bench",
    "--manifest-path",
    "benches/Cargo.toml",
    "--bench",
    "compare_external",
    "--features",
    "compare-external",
]

# mtxdb is swept across its three checksum postures (mdbx/sqlite/fjall have no
# equivalent knob -- see checksum_policy_from_env in compare_external.rs --
# so they run once each). Each tuple is
# (MTXDB_BENCH_EXT_ENGINE value, MTXDB_BENCH_CHECKSUM override or None, the
# row's expected "engine" name). The override is set explicitly for every
# mtxdb invocation, replacing whatever MTXDB_BENCH_CHECKSUM the caller's own
# shell/`.env` may already export, so the sweep always produces exactly
# these three regardless of ambient environment.
INVOCATIONS = (
    ("mtxdb", "disabled", "mtxdb_none"),
    ("mtxdb", "writeonly", "mtxdb_writeonly"),
    ("mtxdb", "full", "mtxdb_full"),
    ("mdbx", None, "mdbx"),
    ("sqlite", None, "sqlite"),
    ("fjall", None, "fjall"),
)
EXPECTED_ENGINES = tuple(row_name for _, _, row_name in INVOCATIONS)


def validate_rows(rows: list[dict]) -> None:
    """Reject partial captures instead of publishing an incomplete comparison."""
    engines = {row["engine"] for row in rows}
    if engines != set(EXPECTED_ENGINES):
        missing = ", ".join(sorted(set(EXPECTED_ENGINES) - engines))
        raise ValueError(f"external capture is incomplete; missing engines: {missing}")
    labels = {row["label"] for row in rows}
    if len(labels) != 1:
        raise ValueError(
            "external capture contains multiple sizes; "
            "run one MTXDB_BENCH_EXT_GB size at a time"
        )


def run_bench() -> None:
    """Run the comparison bench once per invocation, each its own process.

    `/proc/self/smaps_rollup`-based RSS/PSS sampling (see
    `compare_external.rs::smaps_rollup`) reports the whole process, so
    running every engine in one `cargo bench` invocation would let
    allocator retention and a prior engine's still-resident pages bias
    later engines' numbers. `MTXDB_BENCH_EXT_ENGINE` restricts a run to a
    single backend; running it once per row in INVOCATIONS gives each one a
    fresh address space for that measurement. Output from every process is
    concatenated into LATEST so `append_rows` parses it the same as a
    single combined run.
    """
    # compare_external supports a comma-separated curve, but this wrapper
    # publishes one complete engine sweep as one comparison row set. Refuse a
    # multi-size request before doing the expensive work rather than rejecting
    # its output afterwards.
    sizes = os.environ.get("MTXDB_BENCH_EXT_GB", "").split(",")
    if len(sizes) > 1:
        raise SystemExit(
            "MTXDB_BENCH_EXT_GB must specify one size when using "
            "scripts/external_bench.py"
        )
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    with LATEST.open("wb") as out:
        for ext_engine, checksum, row_name in INVOCATIONS:
            env = {**os.environ, "MTXDB_BENCH_EXT_ENGINE": ext_engine}
            if checksum is not None:
                # Always set explicitly (never left to inherit an ambient
                # MTXDB_BENCH_CHECKSUM) so the three-row sweep is
                # deterministic regardless of the caller's own shell/.env.
                # This controls compare_external.rs's frame-level
                # ChecksumPolicy directly (checksum_policy_from_env);
                # "writeonly"/"disabled" additionally relax the checkpoint's
                # own read-time CRC32 verification (MTXDB_CHECKPOINT_CHECKSUM
                # -- see CheckpointChecksumPolicy) to the weakest available
                # posture for that row, since a checkpoint has no "none"
                # tier of its own.
                env["MTXDB_BENCH_CHECKSUM"] = checksum
                env["MTXDB_CHECKPOINT_CHECKSUM"] = (
                    "writeonly" if checksum in ("writeonly", "disabled") else "full"
                )
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
                    f"comparison bench failed for engine={row_name} "
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


def print_table(rows: list[dict], default_run: bool | None = True) -> None:
    """Render every engine side by side from a run's external rows.

    `default_run` distinguishes a plain 0.1 GB run (``True``) from an explicit
    size (``False``). ``None`` means the raw capture came from an earlier run,
    whose environment is unknowable, so no size claim is printed.
    """
    latest: dict[str, dict] = {}
    for row in rows:
        latest[row["engine"]] = row
    # Display names for the row label column; mtxdb's three checksum-policy
    # variants (see INVOCATIONS) get a human label instead of their engine
    # key. Falls back to the raw key for any row this table doesn't know
    # about (an older/newer capture), so it degrades instead of crashing.
    # Collapsing all three to plain "mtxdb" would make two of the five rows
    # indistinguishable: the "crc check mode" column only has two states
    # (checkpoints have no fully-off tier -- "no crc" and "writeonly" both
    # read "writeonly" there), so the row label is the only thing that
    # actually tells those two apart.
    display_names = {
        "mtxdb_none": "mtxdb (no crc)",
        "mtxdb_writeonly": "mtxdb (writeonly)",
        "mtxdb_full": "mtxdb (full crc32)",
        "mdbx": "mdbx",
        "sqlite": "sqlite",
        "fjall": "fjall (lsm)",
    }
    engines = [e for e in EXPECTED_ENGINES if e in latest] or list(latest)
    columns = [m[1] for m in METRICS]

    # Each space-separated word occupies its own header line, keeping units
    # and approximation markers off the metric-name row.
    header_rows = [tuple(label.split(" ")) for label in columns]

    def cell(engine: str, metric: str) -> str:
        value = latest[engine][metric]
        if metric == "checksum":
            return str(value) or "?"
        if metric in ("pss_open_bytes", "pss_warm_bytes") and value is None:
            # `compare_external` emits zero as an unavailable `/proc` PSS
            # sentinel. `external_scenario` turns that into None so it is not
            # rendered or written to history as a bogus 0.0B measurement.
            return "n/a"
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
            return _human_bytes(int(value))
        # mdbx / sqlite keep their B-tree in the DB file's own mapping — the
        # "index" is not a separate resident structure.
        return "in-file"

    values = [[cell(engine, m[0]) for m in METRICS] for engine in engines]
    labels = [display_names.get(engine, engine) for engine in engines]
    label_width = max(len("engine"), *(len(label) for label in labels))
    widths = [
        max(
            *(len(word) for word in header_rows[i]),
            *(len(row[i]) for row in values),
        )
        for i in range(len(columns))
    ]

    print()
    print(
        "".rjust(label_width)
        + "  "
        + "  ".join(header_rows[i][0].rjust(widths[i]) for i in range(len(columns)))
    )
    print(
        "engine".rjust(label_width)
        + "  "
        + "  ".join(header_rows[i][1].rjust(widths[i]) for i in range(len(columns)))
    )
    print(
        "".rjust(label_width)
        + "  "
        + "  ".join(header_rows[i][2].rjust(widths[i]) for i in range(len(columns)))
    )
    print()
    for label, cells in zip(labels, values):
        print(
            f"{label:>{label_width}}  "
            + "  ".join(cell.rjust(widths[i]) for i, cell in enumerate(cells))
        )
    print()
    _footer = ""
    if default_run:
        _footer = (
            "this is the default 0.1 GB run; rerun with MTXDB_BENCH_EXT_GB=0.2 "
            "for a bigger sample"
        )
    elif default_run is False:
        _footer = (
            f"ran with MTXDB_BENCH_EXT_GB= {os.environ.get('MTXDB_BENCH_EXT_GB')} GB"
        )
    print(_footer)
    if any(engine.startswith("mtxdb_") for engine in engines):
        print(
            "mtxdb (no crc): both frame and checkpoint CRC32 off (fastest, "
            "least safe) — mtxdb (writeonly): CRCs written but not "
            "re-verified on read — mtxdb (full crc32): the engine's actual "
            "default, verified on every read. mdbx/sqlite/fjall have no "
            "equivalent read-time checksum of their own to sweep."
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
        raise SystemExit("no `bench: external` rows parsed from the run")
    try:
        validate_rows(scenario.rows)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.append:
        append_rows(scenario)
    # `--no-run` reuses a capture from another invocation. Its size setting
    # is not recorded in the raw output, so do not infer it from this shell.
    default_run = None if args.no_run else "MTXDB_BENCH_EXT_GB" not in os.environ
    print_table(scenario.rows, default_run=default_run)


if __name__ == "__main__":
    main()
