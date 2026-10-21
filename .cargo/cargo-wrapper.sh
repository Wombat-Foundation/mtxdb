#!/usr/bin/env bash
set -e

RUSTC="$1"
shift

CMD=("$RUSTC")

# Codex's sandbox cannot connect to the host sccache daemon.
if [[ -z "$NO_SCCACHE" && -z "$CODEX_THREAD_ID" ]] && command -v sccache >/dev/null 2>&1; then
	CMD=(sccache "$RUSTC")
fi

# Determine the target triple being built. Prefer an explicit `--target`/`-t`
# argument, then the cargo-provided target env vars; an unset target means the
# host. A substring test on the raw args is not enough: it silently injects
# host flags into any other cross target (e.g. aarch64-unknown-linux-gnu,
# x86_64-apple-darwin) and misfires when a path happens to contain "riscv".
HOST_TRIPLE="$(rustc -vV 2>/dev/null | sed -n 's/^host: //p')" || HOST_TRIPLE=""
TARGET_TRIPLE=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
	case "${args[i]}" in
		--target) TARGET_TRIPLE="${args[i + 1]}" ;;
		--target=*) TARGET_TRIPLE="${args[i]#--target=}" ;;
		-t) TARGET_TRIPLE="${args[i + 1]}" ;;
	esac
done
: "${TARGET_TRIPLE:=${RUSTC_TARGET:-${CARGO_BUILD_TARGET:-$HOST_TRIPLE}}}"

# Native-CPU codegen and the mold linker only apply when compiling for the
# host. Cross targets must receive neither: `target-cpu=native` emits host ISA
# features the target cannot execute, and mold is not a valid linker for every
# target (e.g. Mach-O).
#
# A global `RUSTFLAGS="-C target-cpu=native"` (e.g. from a direnv `.env`) is
# applied by cargo to *every* target, including cross targets, and cannot be
# removed by config -- cargo appends RUSTFLAGS to the rustc command line, so
# for cross builds the offending flags are stripped from "$@" here.
#
# If `rustc -vV` fails, HOST_TRIPLE is empty and IS_HOST stays 0: every build
# then takes the cross-filtering path, so mold/target-cpu=native are silently
# omitted. That is the intended fail-safe -- never emit host flags blindly.
IS_HOST=0
if [[ -n "$HOST_TRIPLE" && "$TARGET_TRIPLE" == "$HOST_TRIPLE" ]]; then
	IS_HOST=1
fi

# These flags must come *after* the passthrough args ("$@"), not before: when
# invoked through clippy-driver, $RUSTC is clippy-driver itself and the real
# rustc path is the first element of "$@" -- it must immediately follow, or
# clippy-driver misparses it as an input filename.
if ((IS_HOST)); then
	TARGET_FLAGS=()
	if command -v mold >/dev/null 2>&1; then
		TARGET_FLAGS+=("-C" "link-arg=-fuse-ld=mold")
	fi
	TARGET_FLAGS+=("-C" "target-cpu=native")
	exec "${CMD[@]}" "$@" "${TARGET_FLAGS[@]}"
fi

# Cross target: drop host-only codegen flags that leaked in via RUSTFLAGS,
# in both the split ("-C target-cpu=native") and joined ("-Ctarget-cpu=native")
# forms.
FILTERED=()
while (($#)); do
	case "$1" in
		-C)
			case "${2-}" in
				target-cpu=native | link-arg=-fuse-ld=mold)
					shift 2
					continue
					;;
			esac
			;;
		-Ctarget-cpu=native | -Clink-arg=-fuse-ld=mold)
			shift
			continue
			;;
	esac
	FILTERED+=("$1")
	shift
done
exec "${CMD[@]}" "${FILTERED[@]}"
