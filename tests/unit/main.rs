//! Merged entry point for the ungated integration test files.
//!
//! Each file directly under `tests/` is its own separately compiled and linked
//! test binary. Folding the ungated ones into submodules of this single target
//! keeps `cargo test`/`cargo build --tests` to one link step for them. Files
//! that need a distinct feature set keep their own `[[test]]` entry with
//! `required-features` (`shared_wal_processes` needs `multi-reader`), so the
//! default build neither compiles nor links them.
//!
//! This lives at `tests/unit/main.rs` rather than `tests/unit.rs` so a bare
//! `mod X;` finds each sibling file directly. Cargo does not autodiscover
//! `tests/unit/`, so `Cargo.toml` declares this target explicitly.
//!
//! The `test_*` submodules are public-API-only unit tests relocated out of
//! `src/`; each carries `#![allow(clippy::tests_outside_test_module)]` since it
//! is test code by construction rather than a `#[cfg(test)] mod tests`.

mod index_checkpoint;
mod test_cache;
mod test_csr;
mod test_dag;
mod test_format;
mod test_frontier;
mod test_index_checkpoint;
mod test_matrix_policy;
mod test_packfile_layout;
mod test_redo;
