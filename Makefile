SHELL=/bin/bash
.DEFAULT_GOAL := _help

MAKEFLAGS += --no-print-directory

CARGO ?= cargo
# The bench crate needs a newer compiler than the library's pinned MSRV.
BENCH_TOOLCHAIN ?= 1.90.0

.PHONY: _help
_help:
	@grep -E '^[a-zA-Z_/%-]+:.*?##H' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?##H "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Format, lint, and make docs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.PHONY: all
all: format lint doc test

.PHONY: format check-cargo-sort
format: check-cargo-sort ##H Format code
	prettier -w $$(git ls-files '*.md' '*.y*ml' '*.json')
	pre-commit run --all-files
	-isort $$(git ls-files '*.py')
	-ruff format $$(git ls-files '*.py')
	-ruff check --fix $$(git ls-files '*.py')
	$(CARGO) sort --workspace --grouped

check-cargo-sort:
	@$(CARGO) sort --version >/dev/null 2>&1 || { \
		echo "error: cargo-sort is required; install it with 'cargo install cargo-sort'" >&2; \
		exit 1; \
	}

.PHONY: check
check: ##H Cargo check (core) and code dupe
	$(CARGO) check  --workspace --all-targets --all-features
	jscpd $$(git ls-files '*.rs')


.PHONY: macro
macro: ##H See macro expansion costs
	set -o pipefail; \
	RUSTUP_TOOLCHAIN=nightly $(CARGO) rustc -- -Zmacro-stats 2>&1 | awk -f scripts/macro_totals.awk


.PHONY: lint
lint: ##H Run clippy lints across the workspace crates
	$(CARGO) clippy  --workspace --all-targets --all-features -- $(if $(CI),-D warnings)
	@if command -v flake8 >/dev/null 2>&1; then flake8 --max-line-length 88 $$(git ls-files '*.py'); fi

.PHONY: fix
fix: ##H Apply auto-fixes with clippy across workspace crates
	$(CARGO) clippy --fix  --workspace --allow-dirty --allow-staged --allow-no-vcs --all-targets --all-features


.PHONY: doc
doc: ##H Build docs
	$(CARGO) test --workspace --doc
	$(CARGO) doc --workspace --no-deps
	$(CARGO) rustdoc -p mtxdb-cli --lib -- --document-private-items
	echo '<!doctype html><title>mtxdb documentation</title><h1>mtxdb documentation</h1><ul><li><a href="mtxdb/index.html">mtxdb library</a></li><li><a href="mtxdb_cli/index.html">mtxdb-cli</a></li></ul>' > target/doc/index.html


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test & bench
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Cargo profile for `make test` / `make cov`. Defaults to Cargo's test profile;
# local `.env` may override it (e.g. dev-quick).
MTXDB_TEST_PROFILE ?= test
ifneq ($(strip $(MTXDB_TEST_PROFILE)),)
MTXDB_TEST_PROFILE_FLAG := --profile $(MTXDB_TEST_PROFILE)
endif

# Extra arguments for the test binaries (passed after `--`). For per-test
# durations, set `-Z unstable-options --report-time` in `.env`; each result line
# then ends with `<seconds>`. libtest gates that flag behind RUSTC_BOOTSTRAP, so
# it is set for this recipe only, and only when a `-Z` argument is present.
MTXDB_TEST_ARGS ?=
MTXDB_TEST_ARGS_FLAG := $(if $(strip $(MTXDB_TEST_ARGS)),-- $(MTXDB_TEST_ARGS))
MTXDB_TEST_ENV := $(if $(findstring -Z,$(MTXDB_TEST_ARGS)),RUSTC_BOOTSTRAP=1)

.PHONY: test
test: ##H Run library and workspace tests
	$(MTXDB_TEST_ENV) $(CARGO) test $(MTXDB_TEST_PROFILE_FLAG) --workspace --all-features --lib --tests --timings $(MTXDB_TEST_ARGS_FLAG)

# Drop the Regions/Branches columns from the per-file terminal summary.
LLVM_COV_FLAGS ?= -show-region-summary=false -show-branch-summary=false

.PHONY: cov
cov: ##H Run code coverage and generate HTML report
	# TODO: include `src/bin/` in coverage
	# Run coverage
	$(CARGO) llvm-cov $(MTXDB_TEST_PROFILE_FLAG) \
		-p mtxdb --lib --tests \
		--html --output-dir .coverage \
		--ignore-filename-regex 'src/bin/.*|scripts/.*'
	# Print per-file summary to the terminal (functions/lines only)
	@echo ''
	@echo '══════════════ COVERAGE SUMMARY ══════════════'
	LLVM_COV_FLAGS="${LLVM_COV_FLAGS}" $(CARGO) llvm-cov report $(MTXDB_TEST_PROFILE_FLAG) -p mtxdb \
		--ignore-filename-regex 'src/bin/.*|scripts/.*'
	# Process report to codecov-compatible JSON
	$(CARGO) llvm-cov report $(MTXDB_TEST_PROFILE_FLAG) -p mtxdb \
		--ignore-filename-regex 'src/bin/.*|scripts/.*' \
		--codecov --output-path .coverage/codecov.json
	# JSON summary for the coverage gate (`make cov-check`)
	$(CARGO) llvm-cov report $(MTXDB_TEST_PROFILE_FLAG) -p mtxdb \
		--ignore-filename-regex 'src/bin/.*|scripts/.*' \
		--json --output-path .coverage/summary.json
	@echo DONE. You may open it with:
	@echo firefox .coverage/html/index.html


.PHONY: cov-check
cov-check: ##H Fail if production coverage drops below the floors in scripts/check_coverage.py (run `make cov` first)
	python3 scripts/check_coverage.py .coverage/summary.json

.PHONY: bench
bench: ##H Run benchmarks and append results to the CSV history in benches/csv/
	set -o pipefail; RUSTUP_TOOLCHAIN=$(BENCH_TOOLCHAIN) $(CARGO) bench --manifest-path benches/Cargo.toml --benches --all-targets | tee benches/csv/latest.txt
	python3 scripts/compare_bench.py --current benches/csv/latest.txt \
		--best benches/csv/best.json --out benches/csv/best.json \
		--machine "$$(cat benches/csv/machine.txt 2>/dev/null || hostname)" \
		--csv-dir benches/csv

.PHONY: _bench/external
_bench/external:
	python3 scripts/external_bench.py --append


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Build, install, & clean
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.PHONY: build
build: ##H Build all
	$(CARGO) build --release --timings
	$(CARGO) build --release --timings --manifest-path mtxdb-cli/Cargo.toml


MTXDB_INSTALL_PROFILE ?= release

.PHONY: install
install:	##H Install CLI from source
	$(CARGO) install --profile $(MTXDB_INSTALL_PROFILE) --timings --locked --path mtxdb-cli


.PHONY: clean
clean: ##H Clean build artifacts
	$(CARGO) clean
	cd mtxdb-cli && $(CARGO) clean
	cd benches && $(CARGO) clean
	rm -rf .mypy_cache/ .ruff_cache/ __pycache__/
	rm -rf .coverage/ lcov.info


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Execute command for reach submodule
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# PROJECT_CRATES ?= mtxdb-cli/
PROJECT_CRATES ?= mtxdb-cli/ benches/

.PHONY: sub
sub:	##H Run a command for each crate (set c)
	@test -n "${c}" || (echo "error: set c=<command>"; exit 1)
	@for d in $(PROJECT_CRATES); do \
		echo "--- $$d ---"; \
		if [ "$$d" = "benches/" ]; then \
			(cd $$d && RUSTUP_TOOLCHAIN=$(BENCH_TOOLCHAIN) ${c}) || exit 1; \
		else \
			(cd $$d && ${c}) || exit 1; \
		fi; \
	done


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Extras
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.PHONY: extras/cloc
extras/cloc:	##H Count lines of code for the HEAD revision
	cloc --git HEAD --fmt=2
