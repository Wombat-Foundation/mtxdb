//! Opt-in operation timings for diagnosing slow unit tests.
//!
//! - `MTXDB_TEST_TRACE_MS=<n>` prints each traced operation that took at least
//!   `n` milliseconds, as it finishes.
//! - `MTXDB_TEST_TRACE_SUMMARY=1` also counts every traced operation, however
//!   fast, and prints one `mtxdb-test-io-summary` line per operation when a
//!   thread ends: the call count, the accumulated time and the longest call.
//!   libtest runs each test on its own thread, so the line carries the test's
//!   name. Spans nest (`pool.sync_all` contains pack fsyncs) and run on
//!   background threads too, so the totals are not additive.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

struct Config {
    threshold: Option<Duration>,
    summary: bool,
}

fn config() -> &'static Config {
    static CONFIG: OnceLock<Config> = OnceLock::new();
    CONFIG.get_or_init(|| Config {
        threshold: std::env::var("MTXDB_TEST_TRACE_MS").ok().map(|value| {
            Duration::from_millis(value.parse().expect("MTXDB_TEST_TRACE_MS must be a u64"))
        }),
        summary: std::env::var_os("MTXDB_TEST_TRACE_SUMMARY").is_some_and(|value| value != "0"),
    })
}

/// Per-thread totals, printed when the thread ends.
#[derive(Default)]
struct Totals {
    thread: String,
    by_operation: BTreeMap<&'static str, (u64, Duration, Duration)>,
}

impl Totals {
    fn add(&mut self, operation: &'static str, elapsed: Duration) {
        if self.thread.is_empty() {
            self.thread = std::thread::current()
                .name()
                .unwrap_or("unnamed")
                .to_owned();
        }
        let entry =
            self.by_operation
                .entry(operation)
                .or_insert((0, Duration::ZERO, Duration::ZERO));
        entry.0 = entry.0.saturating_add(1);
        entry.1 = entry.1.saturating_add(elapsed);
        entry.2 = entry.2.max(elapsed);
    }
}

impl Drop for Totals {
    fn drop(&mut self) {
        for (operation, (count, total, longest)) in &self.by_operation {
            eprintln!(
                "mtxdb-test-io-summary pid={} thread={} op={} n={} total_ms={:.1} max_ms={:.1}",
                std::process::id(),
                self.thread,
                operation,
                count,
                total.as_secs_f64() * 1000.0,
                longest.as_secs_f64() * 1000.0
            );
        }
    }
}

thread_local! {
    static TOTALS: RefCell<Totals> = RefCell::new(Totals::default());
}

pub(crate) struct Span<'a> {
    started: Option<Instant>,
    operation: &'static str,
    path: &'a Path,
}

pub(crate) fn span<'a>(operation: &'static str, path: &'a Path) -> Span<'a> {
    let config = config();
    Span {
        started: (config.threshold.is_some() || config.summary).then(Instant::now),
        operation,
        path,
    }
}

impl Drop for Span<'_> {
    fn drop(&mut self) {
        let Some(started) = self.started else {
            return;
        };
        let elapsed = started.elapsed();
        let config = config();
        if config.summary {
            // A span can end during thread teardown, when the totals are gone.
            let _ = TOTALS.try_with(|totals| totals.borrow_mut().add(self.operation, elapsed));
        }
        if config
            .threshold
            .is_some_and(|threshold| elapsed >= threshold)
        {
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
