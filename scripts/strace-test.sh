#!/usr/bin/env bash
# Syscall-time breakdown for ONE test (never the whole suite).
#
#   scripts/strace-test.sh <exact test name>
#   scripts/strace-test.sh database::tests::a_lagging_pool_is_checkpointed_for_the_caller_before_the_segment_fills
#
# Finds the dev-quick test binary that lists the test (build them first with
# `make test`), runs that one test under `strace -c -w` at low priority and
# writes $TMPDIR/strace-<test>.txt: calls and wall time per syscall. That covers
# the fsyncs, renames and truncates that carry no trace span. It runs slower
# under strace. Override the profile directory with MTXDB_STRACE_PROFILE_DIR;
# set MTXDB_STRACE_DRY_RUN=1 to only print which binary it would run, or
# MTXDB_STRACE_TRACE=1 for a per-file fsync/rename breakdown (<test>.files.txt).
set -euo pipefail

test_name="${1:?usage: scripts/strace-test.sh <exact test name>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deps="${MTXDB_STRACE_PROFILE_DIR:-$root/target/dev-quick}/deps"

found=
for candidate in "$deps"/mtxdb-*; do
  # Skip dep-info files and anything not executable; keep the newest match,
  # since older builds of the same crate leave stale binaries behind.
  [[ $candidate == *.d || ! -x $candidate ]] && continue
  # Capture the list first: `--list | grep -q` makes grep exit at the match, the
  # binary gets SIGPIPE, and `pipefail` then reports the match as a failure.
  listing="$("$candidate" --list 2>/dev/null || true)"
  grep -qxF "$test_name: test" <<<"$listing" || continue
  if [ -z "$found" ] || [ "$candidate" -nt "$found" ]; then
    found="$candidate"
  fi
done

if [ -z "$found" ]; then
	echo "no test binary under $deps lists: $test_name" >&2
	exit 1
fi

if [ -n "${MTXDB_STRACE_DRY_RUN:-}" ]; then
  echo "would run: $found --exact $test_name --test-threads=1"
  exit 0
fi

out="${TMPDIR:-/tmp}/strace-${test_name##*::}.txt"
if [ -n "${MTXDB_STRACE_TRACE:-}" ]; then
  # Per-file view: which paths take the fsyncs, fdatasyncs and renames, and how
  # long they wait. Names are normalised (hex ids and digits become H and N).
  raw="${out%.txt}.trace"
  nice -n 19 strace -f -y -T -e trace=fsync,fdatasync,rename,renameat2 -o "$raw" \
    "$found" --exact "$test_name" --test-threads=1
  python3 - "$raw" "${out%.txt}.files.txt" <<'PY'
import collections, re, sys
raw, out = sys.argv[1], sys.argv[2]
stats = collections.defaultdict(lambda: [0, 0.0])
for line in open(raw, errors="replace"):
    m = re.search(r"\b(fsync|fdatasync)\((\d+)<([^>]*)>\).*<([\d.]+)>$", line)
    if m:
        op, path, secs = m.group(1), m.group(3), float(m.group(4))
    else:
        m = re.search(r'\b(rename|renameat2)\(.*?"([^"]+)", .*"([^"]+)".*<([\d.]+)>$', line)
        if not m:
            continue
        op, path, secs = m.group(1), m.group(3), float(m.group(4))
    name = path.rsplit("/", 1)[-1] or path.rstrip("/").rsplit("/", 1)[-1]
    name = re.sub(r"[0-9a-f]{8,}", "H", name)
    name = re.sub(r"\d+", "N", name)
    if path.endswith("/") or "." not in name and op in ("fsync", "fdatasync"):
        name += " (dir?)"
    key = (op, name)
    stats[key][0] += 1
    stats[key][1] += secs
rows = sorted(stats.items(), key=lambda kv: -kv[1][1])
with open(out, "w") as f:
    for (op, name), (n, secs) in rows:
        f.write(f"{secs:8.3f}s  n={n:5d}  {op:9s} {name}\n")
print(f"wrote {out}")
PY
  exit 0
fi

nice -n 19 strace -f -c -w -o "$out" "$found" --exact "$test_name" --test-threads=1
echo "wrote $out"
