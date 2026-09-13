import argparse
import csv
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# The compression benchmark prints one stable row for each payload kind and
# size. Its zstd time is the cost this job tracks; lower is better.
ROW = re.compile(
    r"^(?P<kind>HAMT-shaped|Matrix-JSON-like|incompressible)\s+"
    r"(?P<size>\d+) B\s+raw\s+[\d.]+ us.*?zstd\s+(?P<micros>[\d.]+) us",
    re.MULTILINE,
)

# `benches/compression.rs` emits precisely these cases. Keeping the expected
# set here prevents a first run (whose baseline is `{}`) from accepting a
# truncated benchmark output and publishing only a partial baseline.
EXPECTED_METRICS = {
    f"compression/{kind}/{size}B/zstd_us"
    for kind in ("HAMT-shaped", "Matrix-JSON-like", "incompressible")
    for size in (64, 256, 1_024, 4_096)
}

# One-shot open sweep (`benches/storage.rs::run_open_size_sweep`). The sweep
# is env-sized, so the label set varies; the default size must always be
# present when the family is, and any extra sizes are compared leniently.
DEFAULT_OPEN_LABELS = {"0.1"}

# Three-way comparison bench (`benches/compare_external.rs`), same default
# size, gated behind the `compare-external` feature (absent -> family absent).
DEFAULT_EXT_LABELS = {"0.1"}

ROW_OPEN = re.compile(
    r"^bench: open L=(?P<label>[\d.]+)gb N=\d+ COLS=\d+ WRITE_MS=(?P<write>[\d.]+) "
    r"PACK=(?P<pack>\d+) INDEX=(?P<index>\d+) "
    r"WARM_OPEN_US=(?P<warm_open>[\d.]+) WARM_LOOKUP_US=(?P<warm_lookup>[\d.]+) "
    r"EVICTED_OPEN_US=(?P<evict_open>[\d.]+) "
    r"EVICTED_LOOKUP_US=(?P<evict_lookup>[\d.]+)",
    re.MULTILINE,
)

ROW_EXT = re.compile(
    r"^bench: external ENG=(?P<eng>\w+) L=(?P<label>[\d.]+)gb N=\d+ "
    r"WRITE_MS=(?P<write>[\d.]+) WARM_OPEN_MS=(?P<warm_open>[\d.]+) "
    r"COLD_OPEN_MS=(?P<cold_open>[\d.]+) LOOKUP_US=(?P<lookup>[\d.]+) "
    r"APPEND=\d+ APPEND_MS=(?P<append>[\d.]+) "
    r"APPEND_PUTS_MS=(?P<append_puts>[\d.]+) APPEND_SYNC_MS=(?P<append_sync>[\d.]+)"
    r"(?: APPEND_LOOP_MS=(?P<append_loop>[\d.]+))?"
    r"(?: APPEND_SYNC_ALL_MS=(?P<append_sync_all>[\d.]+))?"
    r" FILES=(?P<files>\d+) "
    r"MEM=(?P<mem>\d+) MEM_LABEL=(?P<mem_label>\w+)",
    re.MULTILINE,
)


# Storage-scenario rows (`benches/storage.rs`). Each emits one machine row per
# scenario (locality), per intent axis (graph/state/timeline), per swarm
# mode/target combination, and per repack events/interval point. The default
# parameter set is what `main()` always runs; when a family is present it must
# cover those defaults so a truncated capture is never published as complete.
DEFAULT_LOCALITY_LABELS = {"small", "medium", "large", "pressure"}
DEFAULT_INTENT_EVENTS = {"20000"}
DEFAULT_SWARM_COMBOS = {
    ("get_many", "adversarial"),
    ("get_many", "organic"),
    ("naive", "adversarial"),
    ("naive", "organic"),
}
DEFAULT_REPACK_POINTS = {("10000", "1000"), ("20000", "1000")}

ROW_LOCALITY = re.compile(
    r"^bench: locality L=(?P<label>\w+) N=\d+ CACHE=\d+ "
    r"PACK_BYTES=(?P<pack>\d+) WRITE_EVENTS_PER_SEC=(?P<write_eps>[\d.]+) "
    r"READ_SYSCALLS=(?P<read_syscalls>\d+) DISK_READ_BYTES=(?P<disk_reads>\d+) "
    r"INDEX_LOSS_PCT=(?P<index_loss>[\d.]+) COLD_GETS_PER_SEC=(?P<cold_gets>[\d.]+) "
    r"WARM_HIT_PCT=(?P<warm_hit>[\d.]+) WARM_GETS_PER_SEC=(?P<warm_gets>[\d.]+)",
    re.MULTILINE,
)

ROW_INTENT = re.compile(
    r"^bench: intent EVENTS=(?P<events>\d+) "
    r"GRAPH_CALLS=(?P<graph_calls>\d+) STATE_CALLS=(?P<state_calls>\d+) "
    r"TIMELINE_CALLS=(?P<timeline_calls>\d+) GRAPH_BYTES=(?P<graph_bytes>\d+) "
    r"STATE_BYTES=(?P<state_bytes>\d+) TIMELINE_BYTES=(?P<timeline_bytes>\d+)",
    re.MULTILINE,
)

ROW_SWARM = re.compile(
    r"^bench: swarm HISTORY=(?P<history>\d+) SWARM=(?P<swarm>\d+) "
    r"MODE=(?P<mode>\w+) TARGET=(?P<target>\w+) FOUND=(?P<found>\d+)/(?P<total>\d+) "
    r"ELAPSED_US=(?P<elapsed>[\d.]+) SYSCALLS=(?P<syscalls>\d+) "
    r"DISK_READ_BYTES=(?P<disk>\d+) EVICTED=(?P<evicted>\w+)",
    re.MULTILINE,
)

ROW_REPACK = re.compile(
    r"^bench: repack EVENTS=(?P<events>\d+) INTERVAL=(?P<interval>\d+) "
    r"RUNS=(?P<runs>\d+) TOTAL_MS=(?P<total>[\d.]+) WRITE_MS=(?P<write>[\d.]+) "
    r"REPACK_MS=(?P<repack>[\d.]+)",
    re.MULTILINE,
)


def load_json(path: Path) -> dict[str, float]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read benchmark baseline {path}: {error}") from error

    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, (int, float))
        for key, value in data.items()
    ):
        raise ValueError(
            f"benchmark baseline {path} must be an object of numeric metrics"
        )
    return data


def open_metrics(output: str) -> dict[str, float]:
    return {
        name: float(value)
        for match in ROW_OPEN.finditer(output)
        for name, value in (
            (f"open/{match['label']}gb/write_ms", match["write"]),
            (f"open/{match['label']}gb/pack_bytes", match["pack"]),
            (f"open/{match['label']}gb/index_bytes", match["index"]),
            (f"open/{match['label']}gb/warm_open_us", match["warm_open"]),
            (f"open/{match['label']}gb/warm_lookup_us", match["warm_lookup"]),
            (f"open/{match['label']}gb/evicted_open_us", match["evict_open"]),
            (f"open/{match['label']}gb/evicted_lookup_us", match["evict_lookup"]),
        )
    }


def external_metrics(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for match in ROW_EXT.finditer(output):
        base = f"ext/{match['eng']}/{match['label']}gb/"
        metrics[base + "write_ms"] = float(match["write"])
        metrics[base + "warm_open_ms"] = float(match["warm_open"])
        metrics[base + "cold_open_ms"] = float(match["cold_open"])
        metrics[base + "lookup_us"] = float(match["lookup"])
        metrics[base + "append_ms"] = float(match["append"])
        metrics[base + "append_puts_ms"] = float(match["append_puts"])
        metrics[base + "append_sync_ms"] = float(match["append_sync"])
        metrics[base + "files_bytes"] = float(match["files"])
        metrics[base + f"bytes_{match['mem_label']}"] = float(match["mem"])
        for name, group in (
            ("append_loop_ms", "append_loop"),
            ("append_sync_all_ms", "append_sync_all"),
        ):
            value = match[group]
            if value is not None:
                metrics[base + name] = float(value)
    return metrics


def storage_metrics(output: str) -> dict[str, float]:
    """Parse the storage-scenario `bench:` rows (locality / intent / swarm /
    repack). Metric names carry the scenario parameters in the path, matching
    the long-format CSV schema without needing extra columns."""
    metrics: dict[str, float] = {}

    for m in ROW_LOCALITY.finditer(output):
        base = f"locality/{m['label']}/"
        metrics[base + "pack_bytes"] = float(m["pack"])
        metrics[base + "write_events_per_sec"] = float(m["write_eps"])
        metrics[base + "read_syscalls"] = float(m["read_syscalls"])
        metrics[base + "disk_read_bytes"] = float(m["disk_reads"])
        metrics[base + "index_loss_pct"] = float(m["index_loss"])
        metrics[base + "cold_gets_per_sec"] = float(m["cold_gets"])
        metrics[base + "warm_hit_pct"] = float(m["warm_hit"])
        metrics[base + "warm_gets_per_sec"] = float(m["warm_gets"])

    for m in ROW_INTENT.finditer(output):
        base = f"intent/{m['events']}/"
        for metric, group in (
            ("graph_calls", "graph_calls"),
            ("state_calls", "state_calls"),
            ("timeline_calls", "timeline_calls"),
            ("graph_bytes", "graph_bytes"),
            ("state_bytes", "state_bytes"),
            ("timeline_bytes", "timeline_bytes"),
        ):
            metrics[base + metric] = float(m[group])

    for m in ROW_SWARM.finditer(output):
        base = f"swarm/{m['history']}/{m['mode']}/{m['target']}/"
        metrics[base + "found"] = float(m["found"])
        metrics[base + "total"] = float(m["total"])
        metrics[base + "elapsed_us"] = float(m["elapsed"])
        metrics[base + "syscalls"] = float(m["syscalls"])
        metrics[base + "disk_read_bytes"] = float(m["disk"])

    for m in ROW_REPACK.finditer(output):
        base = f"repack/{m['events']}/{m['interval']}/"
        metrics[base + "runs"] = float(m["runs"])
        metrics[base + "total_ms"] = float(m["total"])
        metrics[base + "write_ms"] = float(m["write"])
        metrics[base + "repack_ms"] = float(m["repack"])

    return metrics


def parse_current(path: Path) -> dict[str, float]:
    try:
        output = path.read_text()
    except OSError as error:
        raise ValueError(f"cannot read benchmark output {path}: {error}") from error

    compression = {
        f"compression/{match['kind']}/{match['size']}B/zstd_us": float(match["micros"])
        for match in ROW.finditer(output)
    }
    if not compression:
        raise ValueError(
            "no compression benchmark metrics found in "
            f"{path}; refusing to publish an empty baseline"
        )
    if set(compression) != EXPECTED_METRICS:
        missing = sorted(EXPECTED_METRICS - set(compression))
        unexpected = sorted(set(compression) - EXPECTED_METRICS)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise ValueError(
            f"incomplete compression benchmark output ({'; '.join(details)})"
        )

    open_ = open_metrics(output)
    if open_:
        labels = {name.split("/")[1].removesuffix("gb") for name in open_}
        if not DEFAULT_OPEN_LABELS <= labels:
            raise ValueError(
                "open sweep output omits default sizes "
                + ", ".join(sorted(DEFAULT_OPEN_LABELS - labels))
            )

    ext = external_metrics(output)
    if ext:
        for eng in {name.split("/")[1] for name in ext}:
            eng_labels = {
                name.split("/")[2].removesuffix("gb")
                for name in ext
                if name.startswith(f"ext/{eng}/")
            }
            if not DEFAULT_EXT_LABELS <= eng_labels:
                raise ValueError(
                    f"external comparison output for {eng!r} omits default sizes "
                    + ", ".join(sorted(DEFAULT_EXT_LABELS - eng_labels))
                )

    storage = storage_metrics(output)
    storage_labels = {
        name.split("/")[1] for name in storage if name.startswith("locality/")
    }
    if storage_labels and not DEFAULT_LOCALITY_LABELS <= storage_labels:
        raise ValueError(
            "locality output omits default labels "
            + ", ".join(sorted(DEFAULT_LOCALITY_LABELS - storage_labels))
        )
    intent_events = {
        name.split("/")[1] for name in storage if name.startswith("intent/")
    }
    if intent_events and not DEFAULT_INTENT_EVENTS <= intent_events:
        raise ValueError(
            "intent output omits default event counts "
            + ", ".join(sorted(DEFAULT_INTENT_EVENTS - intent_events))
        )
    swarm_combos = {
        (name.split("/")[2], name.split("/")[3])
        for name in storage
        if name.startswith("swarm/")
    }
    if swarm_combos and not DEFAULT_SWARM_COMBOS <= swarm_combos:
        missing = sorted(DEFAULT_SWARM_COMBOS - swarm_combos)
        raise ValueError(
            "swarm output omits default mode/target combos "
            + ", ".join(f"{m}/{t}" for m, t in missing)
        )
    repack_points = {
        (name.split("/")[1], name.split("/")[2])
        for name in storage
        if name.startswith("repack/")
    }
    if repack_points and not DEFAULT_REPACK_POINTS <= repack_points:
        missing = sorted(DEFAULT_REPACK_POINTS - repack_points)
        raise ValueError(
            "repack output omits default events/interval points "
            + ", ".join(f"{e}@{i}" for e, i in missing)
        )

    return {**compression, **open_, **ext, **storage}


def get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def append_history_csv(path: Path, metrics: dict[str, float]) -> None:
    """Append parsed metrics to a long-format CSV history.

    Schema: `timestamp,git_sha,bench,engine,size_gb,metric_name,value`. One
    row per metric from every family in `metrics` (compression / open / ext /
    locality / intent / swarm / repack); the header is written only when the
    file did not previously exist. Long
    form is deliberate: adding a new metric (e.g. `checkpoint_io_ms` or
    `post_replay_len_ms`) never requires a schema migration.
    """
    is_new = not path.exists()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sha = get_git_sha()
    rows: list[list[str]] = []

    for name, value in sorted(metrics.items()):
        parts = name.split("/")
        family = parts[0]
        rest = parts[1:]
        if family == "ext":
            engine = rest[0]
            size_gb = rest[1].removesuffix("gb") if len(rest) > 1 else ""
            metric = "/".join(rest[2:])
        elif family in ("open",):
            engine = "mtxdb"
            size_gb = rest[0].removesuffix("gb") if rest else ""
            metric = "/".join(rest[1:])
        else:
            engine = "mtxdb"
            size_gb = ""
            metric = "/".join(rest)
        rows.append([ts, sha, family, engine, size_gb, metric, f"{value:.6f}"])

    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(
                [
                    "timestamp",
                    "git_sha",
                    "bench",
                    "engine",
                    "size_gb",
                    "metric_name",
                    "value",
                ]
            )
        for row in rows:
            writer.writerow(row)
    print(f"Appended {len(rows)} metrics to {path}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--best", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument(
        "--csv",
        metavar="PATH",
        help="append parsed metrics as long-format rows to this CSV "
        "(schema: ts, engine, size_gb, metric_name, value; header written "
        "only when the file is new)",
    )
    args = parser.parse_args()
    if not math.isfinite(args.margin) or args.margin < 0:
        parser.error("--margin must be a finite, non-negative number")

    try:
        current = parse_current(Path(args.current))
        best = load_json(Path(args.best))
    except ValueError as error:
        parser.error(str(error))

    if args.csv:
        append_history_csv(Path(args.csv), current)

    # Metrics present in last best but not reproducible by this run. Only the
    # opt-in families can be stale (compression is exact-checked above), e.g.
    # a previous run used a wider size sweep or enabled the external bench.
    # Drop them rather than republishing values that can no longer be compared.
    stale = sorted(set(best) - set(current))
    if stale:
        print(
            "dropping stale best metrics not produced by this run: " + ", ".join(stale),
            file=sys.stderr,
        )

    regressions = []
    updated = {name: value for name, value in best.items() if name in current}
    for name, value in current.items():
        previous = updated.get(name)
        if previous is not None and value > previous * (1 + args.margin):
            regressions.append(f"{name}: {value:.2f} us > {previous:.2f} us")
        updated[name] = min(value, previous) if previous is not None else value

    if regressions:
        print("benchmark regression exceeds allowed margin:", file=sys.stderr)
        print("\n".join(regressions), file=sys.stderr)
        sys.exit(1)

    Path(args.out).write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
    print(f"Compared {len(current)} metrics; wrote {args.out}.")


if __name__ == "__main__":
    main()
