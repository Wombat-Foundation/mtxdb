#!/usr/bin/env bash
#
# Fast driver for find_create.py.
#
# One ripgrep pass keeps only lines whose bytes could be a create event's
# outer object (`"type":"m.room.create"`); a second pass drops the embedded
# copies, which in Synapse DAG exports live inside member events and so carry
# a non-empty `state_key`. find_create.py then confirms the outer JSON object
# -- which grep cannot do, because JSON nesting is not a regular language.
#
# Usage:
#   scripts/find_create.sh [FILE ...]      # default: *.jsonl
#
# Output matches find_create.py: one line per input file. Exit status is 0
# when every input file had a valid create, 1 otherwise.
set -eu

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -eq 0 ]; then
  set -- *.jsonl
fi

rg -n --no-heading --no-messages '"type":"m\.room\.create"' "$@" \
  | rg -v '"state_key":"[^"]' \
  | python3 "$here/find_create.py" --candidates "$@"
