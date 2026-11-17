#!/usr/bin/env bash
# Import generated rooms of increasing size and print one row per run.
# usage: scale.sh <gen-export binary> <mtxdb binary> <work dir> <label:mode:events:users>...
set -u
GEN=$1; BIN=$2; W=$3; shift 3
mkdir -p "$W"
printf '%-12s %7s %6s %8s %8s %8s %8s %6s %7s %7s %5s\n' label events users wall_s data_s hamt_s rss_mb merges hamtrec dbMB segs
for spec in "$@"; do
  IFS=: read -r label mode events users <<<"$spec"
  rm -rf "$W/in" "$W/db"
  "$GEN" --mode "$mode" --events "$events" --users "$users" ${GEN_FLAGS:-} --out "$W/in" 2>/dev/null
  "$BIN" -d "$W/db" init >/dev/null 2>&1
  MTXDB_IMPORT_TIMINGS=1 "$BIN" -d "$W/db" import "$W/in/server-keys.jsonl" "$W/in/room-gen.jsonl" >"$W/log.txt" 2>&1 &
  pid=$!; peak=0
  while kill -0 "$pid" 2>/dev/null; do
    hwm=$(awk '/VmHWM/{print $2}' "/proc/$pid/status" 2>/dev/null || true)
    [ -n "${hwm:-}" ] && [ "$hwm" -gt "$peak" ] && peak=$hwm
    sleep 0.05
  done
  wait "$pid"
  wall=$(grep -oE 'import: wall time [0-9.]+' "$W/log.txt" | tail -1 | grep -oE '[0-9.]+$')
  data=$(grep -oE 'timing: data pass \(inputs and key reconciliation\) [0-9.]+m?s' "$W/log.txt" | grep -oE '[0-9.]+m?s$')
  hamt=$(grep -oE 'timing: state HAMT materialize [0-9.]+(ms|s|µs)' "$W/log.txt" | grep -oE '[0-9.]+(ms|s|µs)$')
  merges=$(grep -hoE 'state merges \([0-9]+\)' "$W/log.txt" | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}')
  hrec=$(grep -oE 'state HAMT: [0-9]+ records' "$W/log.txt" | grep -oE '[0-9]+' | head -1)
  segs=$(find "$W/db" -name 'seg-*.csr' | wc -l)
  dbmb=$(( $(du -sb "$W/db" | cut -f1) / 1048576 ))
  rej=$(grep -hE 'derived rejections' "$W/log.txt" | tail -1)
  printf '%-12s %7s %6s %8s %8s %8s %8s %6s %7s %7s %5s  %s\n' "$label" "$events" "$users" "$wall" "$data" "$hamt" "$((peak/1024))" "$merges" "$hrec" "$dbmb" "$segs" "$rej"
done
rm -rf "$W/in" "$W/db"
