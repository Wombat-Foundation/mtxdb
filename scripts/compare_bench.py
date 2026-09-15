"""Parse `cargo bench` output into typed per-scenario CSVs and keep a
`best.json` regression baseline.

The compression bench is the phase gate: its full metric set must be present
before any baseline is published. Every other scenario family (open, external,
storage, elephant) is lenient and only appears when its `bench:` rows do.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# The compression benchmark prints one stable row for each payload kind and
# size. Its zstd time is the cost this job tracks; lower is better.
ROW = re.compile(
    r"^(?P<kind>HAMT-shaped|Matrix-JSON-like|incompressible)\s+"
    r"(?P<size>\d+) B\s+raw\s+(?P<raw>[\d.]+) us.*?"
    r"zstd\s+(?P<zstd>[\d.]+) us.*?"
    r"slowdown\s+(?P<slowdown>[\d.]+)x\s+stored\s+(?P<stored>[\d.]+)%",
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

# Four-way comparison bench (`benches/compare_external.rs`), same default
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
    r"APPEND_PUTS_MS=(?P<append_puts>[\d.]+) "
    r"APPEND_SYNC_MS=(?P<append_sync>[\d.]+)"
    r"(?: APPEND_LOOP_MS=(?P<append_loop>[\d.]+))?"
    r"(?: APPEND_SYNC_ALL_MS=(?P<append_sync_all>[\d.]+))?"
    r" STEADY_APPEND=\d+ STEADY_APPEND_MS=(?P<steady_append>[\d.]+) "
    r"STEADY_APPEND_PUTS_MS=(?P<steady_append_puts>[\d.]+) "
    r"STEADY_APPEND_SYNC_MS=(?P<steady_append_sync>[\d.]+)"
    r" FILES=(?P<files>\d+) "
    r"MEM=(?P<mem>\d+) MEM_LABEL=(?P<mem_label>\w+) "
    r"RSS_OPEN=(?P<rss_open>\d+) PSS_OPEN=(?P<pss_open>\d+) "
    r"RSS_WARM=(?P<rss_warm>\d+) PSS_WARM=(?P<pss_warm>\d+)"
    r"(?: CACHE_CAPACITY=\d+)?"
    r"(?: CHECKSUM=(?P<checksum>\w+))?",
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

# Stage-1 elephant locality bench (`benches/locality.rs`). Phases A/B/C always
# run; their ingest->repack shard caps identify each phase. Phase D (intra-
# shard compaction prototype) emits the `compact` family as a single row.
DEFAULT_ELEPHANT_MODES = {
    ("32768", "32768"),
    ("default", "default"),
    ("32768", "default"),
}

ROW_LOCALITY = re.compile(
    r"^bench: locality L=(?P<label>\w+) N=\d+ CACHE=\d+ "
    r"PACK_BYTES=(?P<pack>\d+) WRITE_EVENTS_PER_SEC=(?P<write_eps>[\d.]+) "
    r"READ_SYSCALLS=(?P<read_syscalls>\d+|n/a) DISK_READ_BYTES=(?P<disk_reads>\d+|n/a) "
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
    r"ELAPSED_US=(?P<elapsed>[\d.]+) SYSCALLS=(?P<syscalls>\d+|n/a) "
    r"DISK_READ_BYTES=(?P<disk>\d+|n/a) EVICTED=(?P<evicted>\w+)",
    re.MULTILINE,
)

ROW_REPACK = re.compile(
    r"^bench: repack EVENTS=(?P<events>\d+) INTERVAL=(?P<interval>\d+) "
    r"RUNS=(?P<runs>\d+) TOTAL_MS=(?P<total>[\d.]+) WRITE_MS=(?P<write>[\d.]+) "
    r"REPACK_MS=(?P<repack>[\d.]+)",
    re.MULTILINE,
)

# Stage-1 elephant locality rows (`benches/locality.rs`): one `bench: elephant`
# row per phase (A/B/C), plus one `bench: compact` row for the Phase D
# intra-shard compaction prototype.
ROW_ELEPHANT = re.compile(
    r"^bench: elephant MAX=(?P<max>\w+) REPACK=(?P<repack>\w+) "
    r"RECORDS=(?P<records>\d+) COLS=(?P<cols>\d+) "
    r"PACKS_PRE=(?P<packs_pre>\d+) PACKS_POST=(?P<packs_post>\d+) "
    r"SAMPLE=(?P<sample>\d+) "
    r"OPEN_PRE_US=(?P<open_pre>[\d.]+) OPEN_POST_US=(?P<open_post>[\d.]+) "
    r"QUERY_PRE_US=(?P<query_pre>[\d.]+) QUERY_POST_US=(?P<query_post>[\d.]+) "
    r"PACKS_REF_PRE=(?P<packs_ref_pre>\d+) PACKS_REF_POST=(?P<packs_ref_post>\d+) "
    r"SEGMENTS_PRE=(?P<segments_pre>\d+) SEGMENTS_POST=(?P<segments_post>\d+) "
    r"OPEN_DISK_PRE=(?P<open_disk_pre>\d+) "
    r"OPEN_DISK_POST=(?P<open_disk_post>\d+) "
    r"OPEN_SYSCALLS_PRE=(?P<open_syscalls_pre>\d+) "
    r"OPEN_SYSCALLS_POST=(?P<open_syscalls_post>\d+) "
    r"QUERY_DISK_PRE=(?P<query_disk_pre>\d+) "
    r"QUERY_DISK_POST=(?P<query_disk_post>\d+) "
    r"QUERY_SYSCALLS_PRE=(?P<query_syscalls_pre>\d+) "
    r"QUERY_SYSCALLS_POST=(?P<query_syscalls_post>\d+) "
    r"FOUND_PRE=(?P<found_pre>\d+) FOUND_POST=(?P<found_post>\d+) "
    r"REPACK_MS=(?P<repack_ms>[\d.]+) SPEEDUP_X=(?P<speedup>[\d.]+)",
    re.MULTILINE,
)

ROW_COMPACT = re.compile(
    r"^bench: compact RECORDS=(?P<records>\d+) COLS=(?P<cols>\d+) "
    r"SHARDS=(?P<shards>\d+) PACKS_REF_PRE=(?P<packs_ref_pre>\d+) "
    r"PACKS_REF_POST=(?P<packs_ref_post>\d+) SEGMENTS_PRE=(?P<segments_pre>\d+) "
    r"SEGMENTS_POST=(?P<segments_post>\d+) WRITTEN=(?P<written>\d+) "
    r"BYTES=(?P<bytes>\d+) WRITE_MS=(?P<write_ms>[\d.]+) FSYNC_MS=(?P<fsync_ms>[\d.]+)",
    re.MULTILINE,
)


@dataclass
class Scenario:
    """One typed CSV artifact: the scenario's column set plus the rows captured
    by a single benchmark run.

    `tracked` maps `bench/param/metric` flat names to the lower-is-better
    values the regression comparison watches; it is filled while parsing each
    row, so a name always pairs with its own row regardless of how many
    metrics a scenario emits per row.
    """

    filename: str
    columns: list[str]
    rows: list[dict] = field(default_factory=list)
    tracked: dict[str, float] = field(default_factory=dict)
    machine: str = ""

    def with_machine(self, machine: str) -> "Scenario":
        """Attach a compact, stable machine key to this scenario.

        The full descriptor remains in ``benches/csv/machine.txt``.  Repeating
        it in every CSV row makes the history mostly hardware prose and makes
        a one-character hardware update require editing every file.
        """
        normalized = " ".join(machine.split())
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
        self.machine = f"machine-{digest}"
        return self


def load_json(path: Path) -> dict[str, float]:
    """Load the numeric `name -> value` baseline at `path`; `{}` if absent."""
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


def compression_scenario(output: str) -> Scenario:
    """Build the compression scenario from the bench output's
    `RAW <size> B ...` rows."""
    scenario = Scenario(
        filename="compression.csv",
        columns=["kind", "payload_len", "raw_us", "zstd_us", "slowdown", "stored_pct"],
    )
    for m in ROW.finditer(output):
        scenario.tracked[f"compression/{m['kind']}/{m['size']}B/zstd_us"] = float(
            m["zstd"]
        )
        scenario.rows.append(
            {
                "kind": m["kind"],
                "payload_len": int(m["size"]),
                "raw_us": float(m["raw"]),
                "zstd_us": float(m["zstd"]),
                "slowdown": float(m["slowdown"]),
                "stored_pct": float(m["stored"]),
            }
        )
    return scenario


def open_scenario(output: str) -> Scenario:
    """Build the open-sweep scenario from the output's `bench: open` rows."""
    scenario = Scenario(
        filename="open.csv",
        columns=[
            "label",
            "write_ms",
            "pack_bytes",
            "index_bytes",
            "warm_open_us",
            "warm_lookup_us",
            "evicted_open_us",
            "evicted_lookup_us",
        ],
    )
    for m in ROW_OPEN.finditer(output):
        row = {
            "label": m["label"],
            "write_ms": float(m["write"]),
            "pack_bytes": int(m["pack"]),
            "index_bytes": int(m["index"]),
            "warm_open_us": float(m["warm_open"]),
            "warm_lookup_us": float(m["warm_lookup"]),
            "evicted_open_us": float(m["evict_open"]),
            "evicted_lookup_us": float(m["evict_lookup"]),
        }
        base = f"open/{m['label']}gb/"
        for metric in (
            "write_ms",
            "pack_bytes",
            "index_bytes",
            "warm_open_us",
            "warm_lookup_us",
            "evicted_open_us",
            "evicted_lookup_us",
        ):
            scenario.tracked[base + metric] = float(row[metric])
        scenario.rows.append(row)
    return scenario


def external_scenario(output: str) -> Scenario:
    """Build the external-comparison scenario from the output's
    `bench: external` rows."""
    scenario = Scenario(
        filename="external.csv",
        columns=[
            "engine",
            "label",
            "write_ms",
            "warm_open_ms",
            "cold_open_ms",
            "lookup_us",
            "append_ms",
            "append_puts_ms",
            "append_sync_ms",
            "append_loop_ms",
            "append_sync_all_ms",
            "steady_append_ms",
            "steady_append_puts_ms",
            "steady_append_sync_ms",
            "files_bytes",
            "mem_bytes",
            "mem_label",
            "rss_open_bytes",
            "pss_open_bytes",
            "rss_warm_bytes",
            "pss_warm_bytes",
            # mtxdb's frame-level ChecksumPolicy actually in effect:
            # "full"/"writeonly"/"none" (MTXDB_BENCH_CHECKSUM), or "na" for
            # engines with no equivalent read-time check (mdbx, sqlite, fjall) --
            # recorded per row so a comparison across history never has to
            # guess which policy produced it. Captures before this column
            # existed leave it blank.
            "checksum",
        ],
    )
    for m in ROW_EXT.finditer(output):
        row = {
            "engine": m["eng"],
            "label": m["label"],
            "write_ms": float(m["write"]),
            "warm_open_ms": float(m["warm_open"]),
            "cold_open_ms": float(m["cold_open"]),
            "lookup_us": float(m["lookup"]),
            "append_ms": float(m["append"]),
            "append_puts_ms": float(m["append_puts"]),
            "append_sync_ms": float(m["append_sync"]),
            "append_loop_ms": _optional_float(m["append_loop"]),
            "append_sync_all_ms": _optional_float(m["append_sync_all"]),
            "steady_append_ms": float(m["steady_append"]),
            "steady_append_puts_ms": float(m["steady_append_puts"]),
            "steady_append_sync_ms": float(m["steady_append_sync"]),
            "files_bytes": int(m["files"]),
            "mem_bytes": int(m["mem"]),
            "mem_label": m["mem_label"],
            "rss_open_bytes": int(m["rss_open"]),
            # `compare_external` uses zero only as its explicit sentinel when
            # `/proc/self/smaps_rollup` cannot be sampled.  A live benchmark
            # process cannot have a real zero PSS, so preserve that distinction
            # as a missing CSV value rather than publishing a false 0-byte
            # memory measurement or tracking it as a performance datapoint.
            "pss_open_bytes": _optional_pss_bytes(m["pss_open"]),
            "rss_warm_bytes": int(m["rss_warm"]),
            "pss_warm_bytes": _optional_pss_bytes(m["pss_warm"]),
            "checksum": m["checksum"] or "",
        }
        base = f"ext/{m['eng']}/{m['label']}gb/"
        for metric in (
            "write_ms",
            "warm_open_ms",
            "cold_open_ms",
            "lookup_us",
            "append_ms",
            "append_puts_ms",
            "append_sync_ms",
            "append_loop_ms",
            "append_sync_all_ms",
            "steady_append_ms",
            "steady_append_puts_ms",
            "steady_append_sync_ms",
            "files_bytes",
            # `mem_bytes` is deliberately excluded: its meaning (mem_label)
            # differs across engines (resident index vs. on-disk file), so
            # it is not a lower-is-better series across engines. PSS is the
            # one memory number this bench captures the same way for all
            # three, via /proc/self/smaps_rollup, so it is trend-tracked.
            "pss_open_bytes",
            "pss_warm_bytes",
        ):
            if row[metric] is not None:
                scenario.tracked[base + metric] = float(row[metric])
        scenario.rows.append(row)
    return scenario


def storage_scenarios(output: str) -> list[Scenario]:
    """Build the locality/intent/swarm/repack scenarios from their `bench:` rows."""
    scenarios: list[Scenario] = []

    locality = Scenario(
        filename="locality.csv",
        columns=[
            "label",
            "pack_bytes",
            "write_events_per_sec",
            "read_syscalls",
            "disk_read_bytes",
            "index_loss_pct",
            "cold_gets_per_sec",
            "warm_hit_pct",
            "warm_gets_per_sec",
        ],
    )
    for m in ROW_LOCALITY.finditer(output):
        row = {
            "label": m["label"],
            "pack_bytes": int(m["pack"]),
            "write_events_per_sec": float(m["write_eps"]),
            "read_syscalls": _optional_u64(m["read_syscalls"]),
            "disk_read_bytes": _optional_u64(m["disk_reads"]),
            "index_loss_pct": float(m["index_loss"]),
            "cold_gets_per_sec": float(m["cold_gets"]),
            "warm_hit_pct": float(m["warm_hit"]),
            "warm_gets_per_sec": float(m["warm_gets"]),
        }
        base = f"locality/{m['label']}/"
        for metric in (
            "pack_bytes",
            "write_events_per_sec",
            "read_syscalls",
            "disk_read_bytes",
            "index_loss_pct",
            "cold_gets_per_sec",
            "warm_hit_pct",
            "warm_gets_per_sec",
        ):
            if row[metric] is not None:
                locality.tracked[base + metric] = float(row[metric])
        locality.rows.append(row)
    scenarios.append(locality)

    intent = Scenario(
        filename="intent.csv",
        columns=[
            "events",
            "graph_calls",
            "state_calls",
            "timeline_calls",
            "graph_bytes",
            "state_bytes",
            "timeline_bytes",
        ],
    )
    for m in ROW_INTENT.finditer(output):
        row = {
            "events": int(m["events"]),
            "graph_calls": int(m["graph_calls"]),
            "state_calls": int(m["state_calls"]),
            "timeline_calls": int(m["timeline_calls"]),
            "graph_bytes": int(m["graph_bytes"]),
            "state_bytes": int(m["state_bytes"]),
            "timeline_bytes": int(m["timeline_bytes"]),
        }
        base = f"intent/{m['events']}/"
        for metric in (
            "graph_calls",
            "state_calls",
            "timeline_calls",
            "graph_bytes",
            "state_bytes",
            "timeline_bytes",
        ):
            intent.tracked[base + metric] = float(row[metric])
        intent.rows.append(row)
    scenarios.append(intent)

    swarm = Scenario(
        filename="swarm.csv",
        columns=[
            "history",
            "mode",
            "target",
            "found",
            "total",
            "elapsed_us",
            "syscalls",
            "disk_read_bytes",
        ],
    )
    for m in ROW_SWARM.finditer(output):
        row = {
            "history": int(m["history"]),
            "mode": m["mode"],
            "target": m["target"],
            "found": int(m["found"]),
            "total": int(m["total"]),
            "elapsed_us": float(m["elapsed"]),
            "syscalls": _optional_u64(m["syscalls"]),
            "disk_read_bytes": _optional_u64(m["disk"]),
        }
        base = f"swarm/{m['history']}/{m['mode']}/{m['target']}/"
        for metric in (
            "found",
            "total",
            "elapsed_us",
            "syscalls",
            "disk_read_bytes",
        ):
            if row[metric] is not None:
                swarm.tracked[base + metric] = float(row[metric])
        swarm.rows.append(row)
    scenarios.append(swarm)

    repack = Scenario(
        filename="repack.csv",
        columns=["events", "interval", "runs", "total_ms", "write_ms", "repack_ms"],
    )
    for m in ROW_REPACK.finditer(output):
        row = {
            "events": int(m["events"]),
            "interval": int(m["interval"]),
            "runs": int(m["runs"]),
            "total_ms": float(m["total"]),
            "write_ms": float(m["write"]),
            "repack_ms": float(m["repack"]),
        }
        base = f"repack/{m['events']}/{m['interval']}/"
        for metric in ("runs", "total_ms", "write_ms", "repack_ms"):
            repack.tracked[base + metric] = float(row[metric])
        repack.rows.append(row)
    scenarios.append(repack)

    return scenarios


def elephant_scenarios(output: str) -> list[Scenario]:
    """Build the elephant (compaction sweep) and compact scenarios."""
    elephant = Scenario(
        filename="elephant.csv",
        columns=[
            "max_shard",
            "repack_shard",
            "records",
            "cols",
            "packs_pre",
            "packs_post",
            "sample",
            "open_pre_us",
            "open_post_us",
            "query_pre_us",
            "query_post_us",
            "packs_ref_pre",
            "packs_ref_post",
            "segments_pre",
            "segments_post",
            "open_disk_pre",
            "open_disk_post",
            "open_syscalls_pre",
            "open_syscalls_post",
            "query_disk_pre",
            "query_disk_post",
            "query_syscalls_pre",
            "query_syscalls_post",
            "found_pre",
            "found_post",
            "repack_ms",
            "speedup_x",
        ],
    )
    for m in ROW_ELEPHANT.finditer(output):
        row = {
            "max_shard": m["max"],
            "repack_shard": m["repack"],
            "records": int(m["records"]),
            "cols": int(m["cols"]),
            "packs_pre": int(m["packs_pre"]),
            "packs_post": int(m["packs_post"]),
            "sample": int(m["sample"]),
            "open_pre_us": float(m["open_pre"]),
            "open_post_us": float(m["open_post"]),
            "query_pre_us": float(m["query_pre"]),
            "query_post_us": float(m["query_post"]),
            "packs_ref_pre": int(m["packs_ref_pre"]),
            "packs_ref_post": int(m["packs_ref_post"]),
            "segments_pre": int(m["segments_pre"]),
            "segments_post": int(m["segments_post"]),
            "open_disk_pre": int(m["open_disk_pre"]),
            "open_disk_post": int(m["open_disk_post"]),
            "open_syscalls_pre": int(m["open_syscalls_pre"]),
            "open_syscalls_post": int(m["open_syscalls_post"]),
            "query_disk_pre": int(m["query_disk_pre"]),
            "query_disk_post": int(m["query_disk_post"]),
            "query_syscalls_pre": int(m["query_syscalls_pre"]),
            "query_syscalls_post": int(m["query_syscalls_post"]),
            "found_pre": int(m["found_pre"]),
            "found_post": int(m["found_post"]),
            "repack_ms": float(m["repack_ms"]),
            "speedup_x": float(m["speedup"]),
        }
        base = f"elephant/{m['max']}/{m['repack']}/"
        for metric in (
            "open_pre_us",
            "open_post_us",
            "query_pre_us",
            "query_post_us",
            "packs_ref_pre",
            "packs_ref_post",
            "segments_pre",
            "segments_post",
            "open_disk_pre",
            "open_disk_post",
            "open_syscalls_pre",
            "open_syscalls_post",
            "query_disk_pre",
            "query_disk_post",
            "query_syscalls_pre",
            "query_syscalls_post",
            "repack_ms",
        ):
            elephant.tracked[base + metric] = float(row[metric])
        elephant.rows.append(row)

    compact = Scenario(
        filename="compact.csv",
        columns=[
            "records",
            "cols",
            "shards",
            "packs_ref_pre",
            "packs_ref_post",
            "segments_pre",
            "segments_post",
            "written",
            "bytes",
            "write_ms",
            "fsync_ms",
        ],
    )
    for m in ROW_COMPACT.finditer(output):
        row = {
            "records": int(m["records"]),
            "cols": int(m["cols"]),
            "shards": int(m["shards"]),
            "packs_ref_pre": int(m["packs_ref_pre"]),
            "packs_ref_post": int(m["packs_ref_post"]),
            "segments_pre": int(m["segments_pre"]),
            "segments_post": int(m["segments_post"]),
            "written": int(m["written"]),
            "bytes": int(m["bytes"]),
            "write_ms": float(m["write_ms"]),
            "fsync_ms": float(m["fsync_ms"]),
        }
        for metric in (
            "write_ms",
            "fsync_ms",
            "shards",
            "segments_post",
            "packs_ref_post",
        ):
            compact.tracked[f"compact/{metric}"] = float(row[metric])
        compact.rows.append(row)

    return [elephant, compact]


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _optional_u64(value: str) -> int | None:
    """Decode an explicit unavailable kernel-counter sentinel."""
    return None if value == "n/a" else int(value)


def _optional_pss_bytes(value: str) -> int | None:
    """Decode compare_external's explicit unavailable-PSS sentinel."""
    parsed = int(value)
    return None if parsed == 0 else parsed


def parse_current(path: Path) -> list[Scenario]:
    """Parse `path`, requiring the full compression gate first, then every
    scenario family whose rows are present."""
    try:
        output = path.read_text()
    except OSError as error:
        raise ValueError(f"cannot read benchmark output {path}: {error}") from error

    compression = compression_scenario(output)
    actual = set(compression.tracked)
    if not actual:
        raise ValueError(
            "no compression benchmark metrics found in "
            f"{path}; refusing to publish an empty baseline"
        )
    if actual != EXPECTED_METRICS:
        missing = sorted(EXPECTED_METRICS - actual)
        unexpected = sorted(actual - EXPECTED_METRICS)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise ValueError(
            f"incomplete compression benchmark output ({'; '.join(details)})"
        )

    scenarios: list[Scenario] = [compression]

    open_ = open_scenario(output)
    if open_.rows:
        labels = {row["label"] for row in open_.rows}
        missing = sorted(DEFAULT_OPEN_LABELS - labels)
        if missing:
            raise ValueError(
                "open sweep output omits default sizes " + ", ".join(missing)
            )
    scenarios.append(open_)

    ext = external_scenario(output)
    if ext.rows:
        engines = {row["engine"] for row in ext.rows}
        # scripts/external_bench.py's own INVOCATIONS sweep (three mtxdb
        # checksum postures + mdbx + sqlite + fjall) is what produces all six
        # engine rows in one capture, and that script enforces the full set
        # itself via validate_rows() -- see EXPECTED_ENGINES there. This
        # generic parser also has to accept a plain `cargo bench --benches`
        # capture (e.g. `make bench`, run directly against
        # benches/compare_external.rs with MTXDB_BENCH_EXT_ENGINE unset),
        # which runs each backend exactly once and never sweeps mtxdb's
        # checksum knob -- that's a legitimately smaller, still-complete set
        # of rows, not a truncated one. Requiring the full six-engine set
        # unconditionally here rejected that valid shape (see the
        # `incomplete external comparison output` failure `make bench` hit).
        # Per this module's own stated policy above ("every other scenario
        # family is lenient and only appears when its bench: rows do"),
        # only check that whichever engines did appear have complete data --
        # not that every possible engine appeared.
        for eng in engines:
            eng_labels = {row["label"] for row in ext.rows if row["engine"] == eng}
            missing = sorted(DEFAULT_EXT_LABELS - eng_labels)
            if missing:
                raise ValueError(
                    f"external comparison output for {eng!r} omits default sizes "
                    + ", ".join(missing)
                )
    scenarios.append(ext)

    scenarios.extend(storage_scenarios(output))
    scenarios.extend(elephant_scenarios(output))

    labels = {
        row["label"]
        for scenario in scenarios
        if scenario.filename == "locality.csv"
        for row in scenario.rows
    }
    if labels:
        missing = sorted(DEFAULT_LOCALITY_LABELS - labels)
        if missing:
            raise ValueError(
                "locality output omits default labels " + ", ".join(missing)
            )
    events = {
        row["events"]
        for scenario in scenarios
        if scenario.filename == "intent.csv"
        for row in scenario.rows
    }
    if events:
        missing = sorted(DEFAULT_INTENT_EVENTS - {str(e) for e in events})
        if missing:
            raise ValueError(
                "intent output omits default event counts " + ", ".join(missing)
            )
    combos = {
        (row["mode"], row["target"])
        for scenario in scenarios
        if scenario.filename == "swarm.csv"
        for row in scenario.rows
    }
    if combos:
        missing = sorted(DEFAULT_SWARM_COMBOS - combos)
        if missing:
            raise ValueError(
                "swarm output omits default mode/target combos "
                + ", ".join(f"{m}/{t}" for m, t in missing)
            )
    points = {
        (str(row["events"]), str(row["interval"]))
        for scenario in scenarios
        if scenario.filename == "repack.csv"
        for row in scenario.rows
    }
    if points:
        missing = sorted(DEFAULT_REPACK_POINTS - points)
        if missing:
            raise ValueError(
                "repack output omits default events/interval points "
                + ", ".join(f"{e}@{i}" for e, i in missing)
            )
    modes = {
        (row["max_shard"], row["repack_shard"])
        for scenario in scenarios
        if scenario.filename == "elephant.csv"
        for row in scenario.rows
    }
    if modes:
        missing = sorted(DEFAULT_ELEPHANT_MODES - modes)
        if missing:
            raise ValueError(
                "elephant output omits default phase modes "
                + ", ".join(f"{a}->{b}" for a, b in missing)
            )

    return scenarios


def get_git_sha() -> str:
    """Short HEAD sha for the CSV history, or "unknown" outside a git tree."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def append_scenario_csv(directory: Path, scenario: Scenario) -> None:
    """Append one run's rows to this scenario's typed CSV (one file per
    scenario, e.g. `locality.csv`). Each scenario has a fixed column set that
    mirrors exactly what its `bench:` row emits, so a missing metric is a
    missing header, not an extra `metric_name` value. New files get a header
    that prefixes `timestamp,git_sha`; later runs append without rewriting it.
    """
    if not scenario.rows:
        return
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / scenario.filename
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sha = get_git_sha()
    header = ["timestamp", "git_sha", "machine", *scenario.columns]

    # Scenario schemas evolve. Preserve prior benchmark history when a new
    # metric is added by rewriting the header once and leaving the new cells
    # empty in historical rows, rather than appending rows whose columns no
    # longer line up with the header.
    if path.exists():
        with open(path, encoding="utf-8", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
            existing_header = handle.seek(0) or next(csv.reader(handle), [])
        if existing_header != header:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(header)
                for row in existing_rows:
                    writer.writerow([row.get(column, "") for column in header])

    is_new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if is_new:
            writer.writerow(header)
        for row in scenario.rows:
            writer.writerow(
                [ts, sha, scenario.machine, *(row[col] for col in scenario.columns)]
            )
    print(f"Appended {len(scenario.rows)} rows to {path}.")


def tracked_metrics(scenarios: list[Scenario]) -> dict[str, float]:
    """Flatten the lower-is-better tracked metrics into a `name -> value` map
    keyed on the `bench/param/metric` scheme for regression comparison.

    Each scenario parser fills `tracked` while parsing rows, so every name is
    already paired with the value taken from its own row."""

    flat: dict[str, float] = {}
    for scenario in scenarios:
        flat.update(scenario.tracked)
    return flat


# Last segment of a flat `bench/param/metric` name. For these, a larger value
# is better (throughput, hit rate, speedup) and the regression check and the
# best-of-history tracking both invert accordingly.
HIGHER_IS_BETTER = frozenset(
    {
        "write_events_per_sec",
        "warm_gets_per_sec",
        "cold_gets_per_sec",
        "warm_hit_pct",
        "speedup_x",
        "found",
        "total",
    }
)


def _is_higher_better(name: str) -> bool:
    return name.rsplit("/", 1)[-1] in HIGHER_IS_BETTER


def main() -> None:
    """Compare `--current` bench output against `--best`, append CSV history,
    and refresh best-of-history. Regressions warn by default; fail with
    `--strict`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--best", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 on regression instead of warning (for CI). Default: warn, "
        "still append history and refresh best.json so noisy single runs do not "
        "break the workflow.",
    )
    parser.add_argument(
        "--machine",
        metavar="SPEC",
        help="machine/spec fingerprint recorded in each CSV row (default: "
        "git_sha; set per machine, e.g. from `cpu_info.sh` output)",
    )
    parser.add_argument(
        "--csv-dir",
        metavar="DIR",
        help="append parsed metrics to typed per-scenario CSV files in this "
        "directory (one file per scenario: compression.csv, open.csv, ...)",
    )
    args = parser.parse_args()
    if not math.isfinite(args.margin) or args.margin < 0:
        parser.error("--margin must be a finite, non-negative number")

    try:
        scenarios = parse_current(Path(args.current))
        best = load_json(Path(args.best))
    except ValueError as error:
        parser.error(str(error))

    machine = args.machine or get_git_sha()
    for scenario in scenarios:
        scenario.with_machine(machine)

    if args.csv_dir:
        for scenario in scenarios:
            append_scenario_csv(Path(args.csv_dir), scenario)

    current = tracked_metrics(scenarios)

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
        if previous is None:
            updated[name] = value
            continue
        if _is_higher_better(name):
            # A smaller value than best is the regression here.
            worsened = value < previous * (1 - args.margin)
            relation = "<"
            kept = max(value, previous)
        else:
            worsened = value > previous * (1 + args.margin)
            relation = ">"
            kept = min(value, previous)
        if worsened:
            regressions.append(f"{name}: {value:.2f} {relation} {previous:.2f}")
        updated[name] = kept

    if regressions:
        heading = (
            "BLOCKING: benchmark regression exceeds allowed margin"
            if args.strict
            else "warning: benchmark regression exceeds allowed margin"
        )
        print(heading + ":", file=sys.stderr)
        print("\n".join(regressions), file=sys.stderr)
        # Always append history and keep best.json fresh; only --strict gates
        # CI-style runs on regressions. Single-run benchmark noise is common, so
        # the default run just warns and continues.
        if args.strict:
            sys.exit(1)

    Path(args.out).write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Compared {len(current)} metrics; wrote {args.out}.")


if __name__ == "__main__":
    main()
