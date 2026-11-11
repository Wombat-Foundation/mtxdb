//! Opt-in operation timings for diagnosing slow unit tests.
//! Set MTXDB_TEST_TRACE_MS to the minimum elapsed milliseconds to report.

use std::path::Path;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

pub(crate) struct Span<'a> {
    active: Option<(Instant, Duration)>,
    operation: &'static str,
    path: &'a Path,
}

pub(crate) fn span<'a>(operation: &'static str, path: &'a Path) -> Span<'a> {
    static THRESHOLD: OnceLock<Option<Duration>> = OnceLock::new();
    let threshold = THRESHOLD.get_or_init(|| {
        std::env::var("MTXDB_TEST_TRACE_MS").ok().map(|value| {
            Duration::from_millis(value.parse().expect("MTXDB_TEST_TRACE_MS must be a u64"))
        })
    });
    Span {
        active: threshold.map(|threshold| (Instant::now(), threshold)),
        operation,
        path,
    }
}

impl Drop for Span<'_> {
    fn drop(&mut self) {
        if let Some((started, threshold)) = self.active {
            let elapsed = started.elapsed();
            if elapsed >= threshold {
                let thread = std::thread::current();
                eprintln!(
                    "mtxdb-test-io pid={} thread={} op={} elapsed_ms={:.3} path={}",
                    std::process::id(),
                    thread.name().unwrap_or("unnamed"),
                    self.operation,
                    elapsed.as_secs_f64() * 1000.0,
                    self.path.display()
                );
            }
        }
    }
}
