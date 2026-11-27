//! mtxdb's optional structured logging API.
//!
//! Enable the `logging` feature to use the level macros and
//! [`init`](crate::logging::init). Hosts can either call `init` to install
//! mtxdb's compact stderr subscriber, or install their own tracing subscriber.

pub use tracing::{debug, error, info, trace, warn};

/// Installs mtxdb's stderr subscriber, filtered by `RUST_LOG`.
///
/// When `RUST_LOG` is unset or invalid, the subscriber defaults to `info`.
/// Applications that already install a tracing subscriber can skip this and
/// use the macros directly.
///
/// # Errors
/// Returns an error if another global tracing subscriber is already installed.
pub fn init() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).try_init()
}
