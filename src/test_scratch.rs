//! Per-process scratch root for test fixtures.
//!
//! Test helpers used to create `mdb_test_*` directories straight under
//! `temp_dir()` and never remove them, so every run left its fixtures behind:
//! thousands of directories accumulated until a small tmpfs hit its quota.
//! Fixtures now live under one root per test process, and the next test
//! process removes the roots of processes that are gone.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const PREFIX: &str = "mdb_test_run_";

/// `<temp_dir>/mdb_test_run_<pid>`, created on first use. Fixture directories
/// go under it.
///
/// Creating the root first removes the roots left by test processes that have
/// exited (Linux only, judged by `/proc/<pid>`). A live process's root is never
/// touched, except across pid namespaces that share one `TMPDIR`, where its
/// pid is invisible from here.
pub(crate) fn root() -> &'static Path {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let base = std::env::temp_dir();
        sweep_dead_runs(&base);
        let root = base.join(format!("{PREFIX}{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("create test scratch root");
        root
    })
}

#[cfg(target_os = "linux")]
fn sweep_dead_runs(base: &Path) {
    let Ok(entries) = std::fs::read_dir(base) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = name
            .to_str()
            .and_then(|name| name.strip_prefix(PREFIX))
            .and_then(|pid| pid.parse::<u32>().ok())
        else {
            continue;
        };
        if !Path::new(&format!("/proc/{pid}")).exists() {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn sweep_dead_runs(_base: &Path) {}
