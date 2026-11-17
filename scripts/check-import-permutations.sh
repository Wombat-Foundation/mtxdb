#!/usr/bin/env bash
# Differential check for the raw import on a large chunked fixture.
#
# Imports the same events in several file orders, once through the whole-room
# path and once through the direct gate, and requires every exported shard to
# be byte-identical. This extends the small in-tree permutation test to a room
# big enough for the direct path to matter.
#
# usage: check-import-permutations.sh FIXTURE_DIR [MTXDB_BIN]
#
#   FIXTURE_DIR  directory of chunk-NN.jsonl files; the lexically first file
#                must carry the room's create event
#   MTXDB_BIN    mtxdb binary; defaults to a release build in this checkout
#
# Plans re-chunk the same causal sequence in different file boundaries:
# original chunks, merged halves, everything but the last chunk merged, the
# whole tail merged, and one file for the room (a control the direct path
# declines). Event order is preserved because a state group cannot be derived
# for an event that arrives before its ancestors; the in-tree unit test owns
# the adversarial late-parent orders. Every plan runs gate-off and gate-on;
# the gate-on run must take the direct path in at least one multi-file plan.
set -euo pipefail

fixture=${1:?usage: check-import-permutations.sh FIXTURE_DIR [MTXDB_BIN]}
if [ $# -ge 2 ]; then
	bin=$2
elif [ -x "$PWD/mtxdb-cli/target/release/mtxdb" ]; then
	bin=$PWD/mtxdb-cli/target/release/mtxdb
else
	bin=$PWD/target/release/mtxdb
fi
[ -x "$bin" ] || {
	echo "no mtxdb binary at $bin" >&2
	exit 1
}
[ -d "$fixture" ] || {
	echo "no fixture directory at $fixture" >&2
	exit 1
}

mapfile -t chunks < <(find "$fixture" -maxdepth 1 -name 'chunk-*.jsonl' -print | LC_ALL=C sort)
if [ "${#chunks[@]}" -lt 3 ]; then
	echo "need at least three chunk-NN.jsonl files in $fixture" >&2
	exit 1
fi
first=${chunks[0]}
tail=("${chunks[@]:1}")

work=$(mktemp -d "${TMPDIR:-/tmp}/mtxdb-permutations.XXXXXX")
trap 'rm -rf "$work"' EXIT

# Materialize one plan as chunk-00... so the CLI glob order is the plan order.
# Each argument is a comma-separated group of source files merged into one
# chunk; groups concatenate in argument order.
make_plan() {
	local name=$1
	shift
	local dir=$work/$name
	mkdir -p "$dir"
	local index=0
	local group
	for group in "$@"; do
		local out
		out=$dir/$(printf 'chunk-%02d.jsonl' "$index")
		local file
		IFS=',' read -r -a parts <<<"$group"
		for file in "${parts[@]}"; do
			cat "$file" >>"$out"
		done
		index=$((index + 1))
	done
}

joined=$(printf '%s,' "${chunks[@]}")
joined=${joined%,}
tail_joined=$(printf '%s,' "${tail[@]}")
tail_joined=${tail_joined%,}

# shellcheck disable=SC2206 # the fixture paths contain no spaces
make_plan causal "${chunks[@]}"
make_plan halves "${chunks[0]},${chunks[1]}" "${chunks[@]:2}"
make_plan first-three "${chunks[0]},${chunks[1]},${chunks[2]}" "${chunks[@]:3}"
make_plan tail-merged "$first" "$tail_joined"
make_plan single "$joined"

dump_shards() {
	local db=$1
	local out=$2
	: >"$out"
	local shard
	for shard in events state server-info; do
		local id
		while read -r id; do
			printf '### %s %s\n' "$shard" "$id" >>"$out"
			"$bin" export "$id" -d "$db" -t "$shard" 2>/dev/null |
				LC_ALL=C sort >>"$out"
		done < <("$bin" collections -d "$db" -t "$shard" -l 0 2>/dev/null |
			awk 'NR > 1 && $1 ~ /^0x/ {print $1}' | LC_ALL=C sort)
	done
}

baseline=
direct_total=0
for plan in causal halves first-three tail-merged single; do
	for mode in off on; do
		db=$work/db
		"$bin" init -d "$db" >/dev/null
		log=$work/$plan-$mode.log
		if [ "$mode" = on ]; then
			env MTXDB_DIRECT_IMPORT=1 MTXDB_DIRECT_DEBUG=1 \
				"$bin" import "$work/$plan"/*.jsonl -d "$db" >"$log" 2>&1
			direct=$(grep -c 'raw import: direct incremental path' "$log" || true)
			direct=${direct:-0}
			direct_total=$((direct_total + direct))
		else
			"$bin" import "$work/$plan"/*.jsonl -d "$db" >"$log" 2>&1
			direct=0
		fi

		dump=$work/$plan-$mode.dump
		dump_shards "$db" "$dump"
		[ -s "$dump" ] || {
			echo "$plan/$mode produced an empty dump" >&2
			exit 1
		}

		if [ -z "$baseline" ]; then
			baseline=$dump
			echo "baseline: $plan/$mode ($(wc -c <"$dump") bytes)"
		elif ! cmp -s "$baseline" "$dump"; then
			echo "MISMATCH: $plan/$mode differs from the baseline" >&2
			exit 1
		fi
		printf '  %-16s %-3s direct files: %s\n' "$plan" "$mode" "$direct"
		rm -rf "$db"
		[ "$dump" = "$baseline" ] || rm -f "$dump"
	done
done

if [ "$direct_total" -eq 0 ]; then
	echo "the direct path never ran; the fixture does not exercise it" >&2
	exit 1
fi
echo "all plans match; direct path handled $direct_total file(s)"
