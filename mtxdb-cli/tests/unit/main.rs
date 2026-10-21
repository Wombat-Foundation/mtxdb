//! Merged entry point for the ungated integration test files.
//!
//! See the root crate's `tests/unit/main.rs` for the rationale: folding ungated
//! integration files into submodules of one target cuts repeated link steps.
//! Cargo does not autodiscover `tests/unit/`, so `Cargo.toml` declares this
//! target explicitly.

mod meta_cli;
