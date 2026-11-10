#!/usr/bin/env python3
"""Measure the distributions a scaling fixture must preserve.

Reads an `mtxdb-export/v1` directory (room exports + optional server keys) and
writes a JSON profile consumed by `gen_scaling_fixture --distributions`.

The profile is descriptive only: it never changes the export and does not run
the importer. Sizes are UTF-8 bytes of the compact PDU as parsed (the export's
own whitespace is not present in a line-oriented export).

Usage:
  python3 scripts/measure_export_distributions.py EXPORT_DIR --output dist.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def quantiles(values: list[int], points: tuple[float, ...]) -> dict[str, int]:
    if not values:
        return {f"p{int(p * 100)}": 0 for p in points}
    ordered = sorted(values)
    result = {}
    for p in points:
        index = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
        result[f"p{int(p * 100)}"] = ordered[index]
    return result


def degree_counts(values: list[int]) -> dict[str, int]:
    counts = Counter(values)
    return {str(key): counts[key] for key in sorted(counts)}


def analyse_room(path: Path) -> dict:
    header = None
    events = []
    declared = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        tag = record.get("t")
        if tag == "header":
            header = record
        elif tag == "event":
            events.append(record)
        elif tag == "end":
            declared = record.get("events")
    if header is None:
        raise SystemExit(f"{path}: no header")
    if declared is not None and declared != len(events):
        raise SystemExit(f"{path}: end declares {declared}, read {len(events)}")

    ids = {record["event_id"] for record in events}
    prev_degrees = []
    auth_degrees = []
    pdu_bytes = []
    state_events = 0
    types = Counter()
    missing_refs = 0
    total_refs = 0
    events_with_missing_ref = 0
    by_prev_set: Counter[tuple[str, ...]] = Counter()

    for record in events:
        pdu = record["pdu"]
        prev = record.get("prev_events", [])
        auth = record.get("auth_events", [])
        prev_degrees.append(len(prev))
        auth_degrees.append(len(auth))
        types[pdu.get("type", "?")] += 1
        if "state_key" in pdu:
            state_events += 1
        pdu_bytes.append(
            len(json.dumps(pdu, separators=(",", ":"), ensure_ascii=False).encode())
        )
        refs = list(prev) + list(auth)
        total_refs += len(refs)
        missing = sum(1 for ref in refs if ref not in ids)
        missing_refs += missing
        if missing:
            events_with_missing_ref += 1
        by_prev_set[tuple(sorted(prev))] += 1

    # A fork is a set of siblings sharing the same prev set. Count every event
    # that has at least one sibling (excluding the first in each set).
    sibling_events = sum(max(0, count - 1) for count in by_prev_set.values())
    event_count = len(events)
    return {
        "room_id": header.get("room_id"),
        "events": event_count,
        "pdu_bytes": {
            "min": min(pdu_bytes) if pdu_bytes else 0,
            "max": max(pdu_bytes) if pdu_bytes else 0,
            "mean": round(statistics.fmean(pdu_bytes), 1) if pdu_bytes else 0,
            **quantiles(pdu_bytes, (0.5, 0.75, 0.9, 0.99)),
        },
        "prev_degree": degree_counts(prev_degrees),
        "auth_degree": degree_counts(auth_degrees),
        "state_rate": round(state_events / event_count, 6) if event_count else 0.0,
        "sibling_rate": round(sibling_events / event_count, 6) if event_count else 0.0,
        "gap_ref_rate": round(missing_refs / total_refs, 6) if total_refs else 0.0,
        "gap_event_rate": round(events_with_missing_ref / event_count, 6)
        if event_count
        else 0.0,
        "types": dict(types.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rooms = []
    for path in sorted(args.export_dir.glob("room-*.jsonl")):
        rooms.append(analyse_room(path))
    if not rooms:
        raise SystemExit(f"{args.export_dir}: no room-*.jsonl files")

    totals = sum(room["events"] for room in rooms)
    weighted = lambda key: round(  # noqa: E731
        sum(room[key] * room["events"] for room in rooms) / totals, 6
    )
    # Pool the per-event distributions across rooms for the generator.
    pdu_bytes = []
    prev_degrees = []
    auth_degrees = []
    for path in sorted(args.export_dir.glob("room-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("t") != "event":
                continue
            pdu_bytes.append(
                len(
                    json.dumps(
                        record["pdu"], separators=(",", ":"), ensure_ascii=False
                    ).encode()
                )
            )
            prev_degrees.append(len(record.get("prev_events", [])))
            auth_degrees.append(len(record.get("auth_events", [])))

    profile = {
        "source": str(args.export_dir),
        "rooms": len(rooms),
        "events": totals,
        "pdu_bytes": {
            "min": min(pdu_bytes),
            "max": max(pdu_bytes),
            "mean": round(statistics.fmean(pdu_bytes), 1),
            **quantiles(pdu_bytes, (0.5, 0.75, 0.9, 0.99)),
        },
        "prev_degree": degree_counts(prev_degrees),
        "auth_degree": degree_counts(auth_degrees),
        "state_rate": weighted("state_rate"),
        "sibling_rate": weighted("sibling_rate"),
        "gap_ref_rate": weighted("gap_ref_rate"),
        "gap_event_rate": weighted("gap_event_rate"),
        "per_room_events": sorted((room["events"] for room in rooms), reverse=True),
        "room_profiles": rooms,
    }
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(
        f"{profile['rooms']} rooms, {profile['events']} events; "
        f"state={profile['state_rate']:.3f} sibling={profile['sibling_rate']:.3f} "
        f"gap_ref={profile['gap_ref_rate']:.4f} "
        f"pdu p50/p90/p99={profile['pdu_bytes']['p50']}/"
        f"{profile['pdu_bytes']['p90']}/{profile['pdu_bytes']['p99']} B"
    )


if __name__ == "__main__":
    main()
