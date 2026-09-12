//! Build script — emits `GIT_DESCRIBE` from `git describe --always --dirty`.

/// Sets `GIT_DESCRIBE` env var to the output of `git describe --always --dirty`.
fn main() {
    let repo_root = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned());
    let git_describe = std::process::Command::new("git")
        .args(["describe", "--always", "--dirty"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map_or_else(
            || env!("CARGO_PKG_VERSION").to_owned(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_owned(),
        );
    println!("cargo:rustc-env=GIT_DESCRIBE={git_describe}");
    if let Some(repo_root) = repo_root {
        println!("cargo:rerun-if-changed={repo_root}/.git/HEAD");
        println!("cargo:rerun-if-changed={repo_root}/.git/refs/");
    }
}
