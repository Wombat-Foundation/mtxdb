pub(crate) fn vmtouch_on_path() -> bool {
    std::process::Command::new("vmtouch")
        .arg("-h")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}
