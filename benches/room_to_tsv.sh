#!/bin/sh
# Turn a merged room DAG (one event JSON per line) into the edge-layout bench
# input: event_id, prev ids, auth ids, relation target (tab-separated, ids
# comma-separated). Usage: room_to_tsv.sh merged-room.jsonl > room.tsv
tmp=$(mktemp) || exit 1
trap 'rm -f "$tmp"' 0 HUP INT TERM
if jq -r '[.event_id,
        (.prev_events | map(if type == "array" then .[0] else . end) | join(",")),
        (.auth_events | map(if type == "array" then .[0] else . end) | join(",")),
        (.content["m.relates_to"].event_id // "")] | @tsv' "$1" > "$tmp"; then
  awk -F'\t' '!seen[$1]++' "$tmp"
else
  exit 1
fi
  awk -F'\t' '!seen[$1]++'
