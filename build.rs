fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version_cmd = std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .expect("failed to run rustc --version");
    let version = String::from_utf8_lossy(&version_cmd.stdout);
    if version.contains("nightly") || version.contains("dev") {
        println!("cargo:rustc-cfg=coverage_nightly");
    }
}
