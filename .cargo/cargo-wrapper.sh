#!/usr/bin/env bash
set -e

RUSTC="$1"
shift

CMD=("$RUSTC")

# Codex's sandbox cannot connect to the host sccache daemon.
if [[ -z "$NO_SCCACHE" && -z "$CODEX_THREAD_ID" ]] && command -v sccache >/dev/null 2>&1; then
	CMD=(sccache "$RUSTC")
fi

MOLD_ARGS=()
if command -v mold >/dev/null 2>&1; then
	# Do not use mold if cross-compiling to webassembly or riscv (SP1)
	if [[ "$*" == *"wasm32"* ]] || [[ "$*" == *"riscv"* ]]; then
		: # skip mold
	else
		MOLD_ARGS=("-C" "link-arg=-fuse-ld=mold")
	fi
fi

# Our injected flags must come *after* the passthrough args ("$@"), not
# before: when invoked through clippy-driver, $RUSTC is clippy-driver
# itself and the real rustc path is the first element of "$@" -- it must
# immediately follow, or clippy-driver misparses it as an input filename.
exec "${CMD[@]}" "$@" -C target-cpu=native "${MOLD_ARGS[@]}"
