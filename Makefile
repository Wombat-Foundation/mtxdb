SHELL=/bin/bash
.DEFAULT_GOAL := _help

MAKEFLAGS += --no-print-directory

CARGO ?= cargo

.PHONY: _help
_help:
	@grep -E '^[a-zA-Z_/%-]+:.*?##H' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?##H "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'



.PHONY: format
format: check-cargo-sort ##H Format code
	prettier -w $$(git ls-files '*.md' '*.y*ml')
	pre-commit run --all-files
	$(CARGO) sort --workspace --grouped

.PHONY: check-cargo-sort
check-cargo-sort:
	@command -v cargo-sort >/dev/null || { echo "cargo-sort is required; install it with: cargo install cargo-sort" >&2; exit 1; }

.PHONY: check
check: ##H Cargo check (core) and code dupe
	$(CARGO) check --all-targets
	@command -v jscpd >/dev/null || { echo "jscpd is required; install it with: npm install -g jscpd" >&2; exit 1; }
	jscpd $$(git ls-files '*.rs')

.PHONY: lint
lint: ##H Run clippy lints (only core, not full workspace)
	$(CARGO) clippy --all-targets --all-features -- $(if $(CI),-D warnings)

.PHONY: fix
fix: ##H Apply auto-fixes with clippy (only core)
	$(CARGO) clippy --fix --allow-dirty --allow-staged --allow-no-vcs --all-targets



.PHONY: doc
doc: ##H Build docs
	$(CARGO) test --workspace --doc
	$(CARGO) doc --workspace --no-deps
	echo '<meta http-equiv="refresh" content="0;url=mtxdb/index.html">' > target/doc/index.html



.PHONY: test
test: ##H Run tests (only core)
	$(CARGO) test --lib --tests --timings

# Drop the Regions/Branches columns from the per-file terminal summary.
LLVM_COV_FLAGS ?= -show-region-summary=false -show-branch-summary=false

.PHONY: cov
cov: ##H Run code coverage and generate HTML report
	# TODO: include `src/bin/` in coverage
	# Run coverage
	$(CARGO) llvm-cov --lib --tests \
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


.PHONY: build
build: ##H Build all
	$(CARGO) build --release --timings
	$(CARGO) build --release --timings --manifest-path mtxdb-ffi/Cargo.toml
	RUSTFLAGS= $(CARGO) build --release --timings --manifest-path mtxdb-wasm/Cargo.toml --target wasm32-wasip1


.PHONY: bench
bench: ##H Run benchmarks
	$(CARGO) bench --benches

.PHONY: clean
clean: ##H Clean build artifacts
	$(CARGO) clean
	cd mtxdb-ffi && $(CARGO) clean
	cd mtxdb-wasm && $(CARGO) clean
