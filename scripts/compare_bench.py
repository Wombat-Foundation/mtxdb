import argparse
import json
import math
import re
import sys
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
        raise ValueError(f"benchmark baseline {path} must be an object of numeric metrics")
    return data


def parse_current(path: Path) -> dict[str, float]:
    try:
        output = path.read_text()
    except OSError as error:
        raise ValueError(f"cannot read benchmark output {path}: {error}") from error

    metrics = {
        f"compression/{match['kind']}/{match['size']}B/zstd_us": float(match["micros"])
        for match in ROW.finditer(output)
    }
    if not metrics:
        raise ValueError(
            f"no compression benchmark metrics found in {path}; refusing to publish an empty baseline"
        )
    if set(metrics) != EXPECTED_METRICS:
        missing = sorted(EXPECTED_METRICS - set(metrics))
        unexpected = sorted(set(metrics) - EXPECTED_METRICS)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise ValueError(f"incomplete compression benchmark output ({'; '.join(details)})")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--best", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--margin", type=float, default=0.10)
    args = parser.parse_args()
    if not math.isfinite(args.margin) or args.margin < 0:
        parser.error("--margin must be a finite, non-negative number")

    try:
        current = parse_current(Path(args.current))
        best = load_json(Path(args.best))
    except ValueError as error:
        parser.error(str(error))

    missing = sorted(set(best) - set(current))
    if missing:
        parser.error(
            "benchmark output omits metrics present in the best baseline; "
            "stale best values would otherwise be published untouched: "
            + ", ".join(missing)
        )

    regressions = []
    updated = dict(best)
    for name, value in current.items():
        previous = best.get(name)
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
