//! Smoke tests for the `mtxdb meta` command.

#[cfg(test)]
mod tests {
    use std::process::Command;

    #[test]
    fn meta_json_smoke_output_is_valid() {
        let root = std::env::temp_dir().join(format!(
            "mtxdb-meta-cli-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let output = Command::new(env!("CARGO_BIN_EXE_mtxdb"))
            .args(["meta", "sidecars", "--json", "--dir"])
            .arg(&root)
            .output()
            .expect("run mtxdb meta");
        assert!(output.status.success(), "stderr: {:?}", output.stderr);
        let mut json = output.stdout;
        let value = simd_json::to_owned_value(&mut json).expect("meta output must be valid JSON");
        assert!(matches!(value, simd_json::OwnedValue::Array(_)));

        let _ = std::fs::remove_dir_all(root);
    }
}
