//! End-to-end check that `-t` carries the selected pool into execution.

#[cfg(test)]
mod tests {
    use std::process::Command;

    #[test]
    fn server_info_shard_type_selects_the_server_info_pool() {
        let root = std::env::temp_dir().join(format!(
            "mtxdb-shard-type-cli-{}-{}",
            std::process::id(),
            std::thread::current()
                .name()
                .unwrap_or("test")
                .replace(':', "_")
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let init = Command::new(env!("CARGO_BIN_EXE_mtxdb"))
            .args(["init", "--dir"])
            .arg(&root)
            .output()
            .expect("run mtxdb init");
        assert!(init.status.success(), "stderr: {:?}", init.stderr);

        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/mtxdb-export/server-keys-v1.jsonl");
        let import = Command::new(env!("CARGO_BIN_EXE_mtxdb"))
            .args(["-t", "server-info", "import", "--dir"])
            .arg(&root)
            .arg(&fixture)
            .output()
            .expect("run mtxdb import");
        assert!(import.status.success(), "stderr: {:?}", import.stderr);

        let server_info = Command::new(env!("CARGO_BIN_EXE_mtxdb"))
            .args(["-t", "server-info", "collections", "--dir"])
            .arg(&root)
            .output()
            .expect("run mtxdb collections for server-info");
        let server_info_stdout = String::from_utf8_lossy(&server_info.stdout);
        assert!(
            server_info.status.success(),
            "stderr: {}",
            String::from_utf8_lossy(&server_info.stderr)
        );
        assert!(
            server_info_stdout.contains("collection")
                && !server_info_stdout.contains("no collections"),
            "server-info pool should have collections: {server_info_stdout}"
        );

        let default = Command::new(env!("CARGO_BIN_EXE_mtxdb"))
            .args(["collections", "--dir"])
            .arg(&root)
            .output()
            .expect("run mtxdb collections for the default pool");
        let default_stdout = String::from_utf8_lossy(&default.stdout);
        assert!(default.status.success(), "stderr: {:?}", default.stderr);
        assert!(
            default_stdout.contains("no collections found"),
            "the default event pool must stay empty: {default_stdout}"
        );

        let _ = std::fs::remove_dir_all(root);
    }
}
