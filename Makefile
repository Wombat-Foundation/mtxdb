SHELL=/bin/bash
.DEFAULT_GOAL := _help

MAKEFLAGS += --no-print-directory

CARGO ?= cargo

.PHONY: _help
_help:
	@grep -E '^[a-zA-Z_/%-]+:.*?##H' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?##H "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Format, lint, and make docs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.PHONY: all
all: format lint doc fix test install

.PHONY: format
format: ##H Format code
	prettier -w $$(git ls-files '*.md' '*.y*ml' '*.json')
	pre-commit run --all-files
	-isort $$(git ls-files '*.py')
	-ruff format $$(git ls-files '*.py')
	-ruff check --fix $$(git ls-files '*.py')
	$(CARGO) sort --workspace --grouped

.PHONY: check
check: ##H Cargo check (core) and code dupe
	$(CARGO) check  --workspace --all-targets --all-features
	jscpd $$(git ls-files '*.rs')

.PHONY: lint
lint: ##H Run clippy lints (only core, not full workspace)
	$(CARGO) clippy  --workspace --all-targets --all-features -- $(if $(CI),-D warnings)
	-flake8 --max-line-length 88 $$(git ls-files '*.py')

.PHONY: fix
fix: ##H Apply auto-fixes with clippy (only core)
	$(CARGO) clippy --fix  --workspace --allow-dirty --allow-staged --allow-no-vcs --all-targets --all-features


.PHONY: doc
doc: ##H Build docs
	$(CARGO) test --workspace --doc
	# Document library crates only — the mtxdb-cli binary shares the name
	# "mtxdb" with the root facade lib, hitting cargo #6313.  Skip it;
	# CLI usage is covered by `mtxdb --help`.
	$(CARGO) doc -p mtxdb -p mtxdb-core --no-deps
	echo '<meta http-equiv="refresh" content="0;url=mtxdb/index.html">' > target/doc/index.html


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test & bench
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.PHONY: test
test: ##H Run tests (only core)
	$(CARGO) test --workspace --lib --tests --timings

# Drop the Regions/Branches columns from the per-file terminal summary.
LLVM_COV_FLAGS ?= -show-region-summary=false -show-branch-summary=false

.PHONY: cov
cov: ##H Run code coverage and generate HTML report
	# TODO: include `src/bin/` in coverage
	# Run coverage
	$(CARGO) llvm-cov -p mtxdb-core --lib --tests \
		--html --output-dir .coverage \
		--ignore-filename-regex 'src/bin/.*|scripts/.*'
	# Print per-file summary to the terminal (functions/lines only)
	@echo ''
	@echo '══════════════ COVERAGE SUMMARY ══════════════'
	LLVM_COV_FLAGS="${LLVM_COV_FLAGS}" $(CARGO) llvm-cov report \
		--ignore-filename-regex 'src/bin/.*|scripts/.*'
	# Process report to codecov-compatible JSON
	$(CARGO) llvm-cov report \
		--ignore-filename-regex 'src/bin/.*|scripts/.*' \
		--codecov --output-path .coverage/codecov.json
	@echo DONE. You may open it with:
	@echo firefox .coverage/html/index.html


.PHONY: bench
bench: ##H Run benchmarks and append results to the CSV history in benches/csv/
	$(CARGO) bench --benches --all-features --all-targets | tee benches/csv/latest.txt
	python3 scripts/compare_bench.py --current benches/csv/latest.txt \
		--best benches/csv/best.json --out benches/csv/best.json \
		--machine "$$(cat benches/csv/machine.txt 2>/dev/null || hostname)" \
		--csv-dir benches/csv

.PHONY: _bench/external
_bench/external: ##H Comparison bench (mtxdb vs mdbx vs sqlite) -> benches/csv/external.csv + table
	python3 scripts/external_bench.py


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Build, install, & clean
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.PHONY: build
build: ##H Build all
	$(CARGO) build --release --timings
	$(CARGO) build --release --timings --manifest-path mtxdb-cli/Cargo.toml
	$(CARGO) build --release --timings --manifest-path mtxdb-ffi/Cargo.toml
	RUSTFLAGS= $(CARGO) build --release --timings --manifest-path mtxdb-wasm/Cargo.toml --target wasm32-wasip1

.PHONY: install
install:	##H Install CLI from source
	$(CARGO) install --timings --locked --path mtxdb-cli


.PHONY: clean
clean: ##H Clean build artifacts
	$(CARGO) clean
	cd mtxdb-cli && $(CARGO) clean
	cd mtxdb-core && $(CARGO) clean
	cd mtxdb-ffi && $(CARGO) clean
	cd mtxdb-wasm && $(CARGO) clean
	rm -rf .coverage/ lcov.info


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Execute command for reach submodule
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PROJECT_CRATES ?= mtxdb-cli/ mtxdb-core/ mtxdb-ffi/ mtxdb-wasm/

.PHONY: sub
sub:	##H Run a command for each crate (set c)
	@test -n "${c}" || (echo "error: set c=<command>"; exit 1)
	@for d in $(PROJECT_CRATES); do echo "--- $$d ---"; (cd $$d && ${c}) || exit 1; done
