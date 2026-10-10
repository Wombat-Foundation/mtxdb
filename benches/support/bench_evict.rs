use std::path::Path;

/// Best-effort page-cache eviction via `vmtouch -e`.
pub(crate) fn evict_dir(dir: &Path) -> bool {
    std::process::Command::new("vmtouch")
        .arg("-e")
        .arg(dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}
