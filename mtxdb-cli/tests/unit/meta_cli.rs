//! Smoke tests for the `mtxdb meta` command.

#[cfg(test)]
mod tests {
    use std::process::Command;

    use rezzy::json::Value;

    /// A per-test temp root. Test thread names contain `::`, which is not
    /// valid in a Windows path, so colons become underscores.
    fn test_root(prefix: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "{prefix}-{}-{}",
            std::process::id(),
            std::thread::current()
                .name()
                .unwrap_or("test")
                .replace(':', "_")
        ))
    }

    #[test]
    fn meta_json_smoke_output_is_valid() {
        let root = test_root("mtxdb-meta-cli");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let output = Command::new(env!("CARGO_BIN_EXE_mtxdb"))
            .args(["meta", "sidecars", "--json", "--dir"])
            .arg(&root)
            .output()
            .expect("run mtxdb meta");
        assert!(output.status.success(), "stderr: {:?}", output.stderr);
        let value = Value::parse_bytes(&output.stdout).expect("meta output must be valid JSON");
        assert!(matches!(value, Value::Array(_)));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn meta_json_reports_checkpoint_diagnostic() {
        let root = test_root("mtxdb-meta-cli-checkpoint");
        let state = root.join("mtpl-state");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&state).unwrap();
        std::fs::write(state.join("index.checkpoint"), b"corrupt checkpoint").unwrap();

        let output = Command::new(env!("CARGO_BIN_EXE_mtxdb"))
            .args(["meta", "checkpoint", "--json", "--dir"])
            .arg(&root)
            .output()
            .expect("run mtxdb meta checkpoint");
        assert!(output.status.success(), "stderr: {:?}", output.stderr);
        let value = Value::parse_bytes(&output.stdout).expect("meta output must be valid JSON");
        let Value::Array(records) = value else {
            panic!("meta JSON must be an array");
        };
        assert!(records
            .iter()
            .any(|record| { record.get("severity").and_then(Value::as_str) == Some("WARN") }));

        let _ = std::fs::remove_dir_all(root);
    }
}
