//! Raw-import warning output across multiple files.

#[cfg(test)]
mod tests {
    use std::process::Command;

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

    fn with_content_hash(line: &str, room_version: &str) -> String {
        let mut value = rezzy::json::Value::parse_bytes(line.as_bytes()).expect("valid event JSON");
        let hash =
            rezzy::compute_content_hash(&value, room_version).expect("content hash is computable");
        if let rezzy::json::Value::Object(fields) = &mut value {
            let mut hashes = rezzy::json::Object::new();
            hashes.insert("sha256".to_owned(), rezzy::json::Value::String(hash));
            fields.insert("hashes".to_owned(), rezzy::json::Value::Object(hashes));
        }
        rezzy::json::write_string_value(&value).expect("event serializes")
    }

    #[test]
    fn raw_import_rejects_conflicting_auth_projection() {
        let root = test_root("mtxdb-raw-projection-cli");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let first_pl = with_content_hash(
            r#"{"event_id":"$pl","sender":"@alice:test","room_id":"!room:test","type":"m.room.power_levels","state_key":"","content":{"users":{"@alice:test":100},"invite":50},"prev_events":["$join"],"auth_events":["$create","$join"]}"#,
            "10",
        );
        let first = root.join("first.jsonl");
        std::fs::write(
            &first,
            format!(
                "{}\n{}\n{}\n",
                r#"{"event_id":"$create","sender":"@alice:test","room_id":"!room:test","type":"m.room.create","state_key":"","content":{"room_version":"10"},"prev_events":[],"auth_events":[]}"#,
                r#"{"event_id":"$join","sender":"@alice:test","room_id":"!room:test","type":"m.room.member","state_key":"@alice:test","content":{"membership":"join"},"prev_events":["$create"],"auth_events":["$create"]}"#,
                first_pl,
            ),
        )
        .unwrap();
        let second_pl = with_content_hash(
            r#"{"event_id":"$pl","sender":"@alice:test","room_id":"!room:test","type":"m.room.power_levels","state_key":"","content":{"users":{"@alice:test":100},"invite":0},"prev_events":["$join"],"auth_events":["$create","$join"]}"#,
            "10",
        );
        let second = root.join("second.jsonl");
        std::fs::write(&second, format!("{second_pl}\n")).unwrap();

        let init = Command::new(env!("CARGO_BIN_EXE_mtxdb"))
            .args(["init", "--dir"])
            .arg(&root)
            .output()
            .expect("run mtxdb init");
        assert!(
            init.status.success(),
            "init stderr: {}",
            String::from_utf8_lossy(&init.stderr)
        );

        let first_output = Command::new(env!("CARGO_BIN_EXE_mtxdb"))
            .args(["import", "--dir"])
            .arg(&root)
            .arg(&first)
            .output()
            .expect("run first import");
        assert!(
            first_output.status.success(),
            "first import stderr: {}",
            String::from_utf8_lossy(&first_output.stderr)
        );

        let second_output = Command::new(env!("CARGO_BIN_EXE_mtxdb"))
            .args(["import", "--dir"])
            .arg(&root)
            .arg(&second)
            .output()
            .expect("run second import");
        let stderr = String::from_utf8_lossy(&second_output.stderr);
        assert!(
            !second_output.status.success(),
            "a verified projection conflict must fail the import: {stderr}"
        );
        assert!(
            stderr.contains("auth projection for event $pl conflicts with the stored projection"),
            "the second import must name the conflicting projection: {stderr}"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn raw_import_warnings_only_name_current_file_events() {
        let root = test_root("mtxdb-raw-warning-cli");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let first = root.join("first.jsonl");
        std::fs::write(
            &first,
            concat!(
                r#"{"event_id":"$create","sender":"@alice:test","room_id":"!room:test","type":"m.room.create","state_key":"","content":{"room_version":"11"},"prev_events":[],"auth_events":[]}"#,
                "\n",
                r#"{"event_id":"$join","sender":"@alice:test","room_id":"!room:test","type":"m.room.member","state_key":"@alice:test","content":{"membership":"join"},"prev_events":["$create"],"auth_events":["$create"]}"#,
                "\n",
                // Mallory is not joined, so check 5 rejects this event. The
                // source does not claim it is rejected.
                r#"{"event_id":"$bad","sender":"@mallory:test","room_id":"!room:test","type":"m.room.join_rules","state_key":"","content":{"join_rule":"public"},"prev_events":["$join"],"auth_events":["$create","$join"]}"#,
                "\n",
            ),
        )
        .unwrap();
        let second = root.join("second.jsonl");
        std::fs::write(
            &second,
            concat!(
                r#"{"event_id":"$live","sender":"@alice:test","room_id":"!room:test","type":"m.room.message","content":{},"prev_events":["$bad"],"auth_events":["$create","$join"]}"#,
                "\n",
            ),
        )
        .unwrap();

        let init = Command::new(env!("CARGO_BIN_EXE_mtxdb"))
            .args(["init", "--dir"])
            .arg(&root)
            .output()
            .expect("run mtxdb init");
        assert!(
            init.status.success(),
            "init stderr: {}",
            String::from_utf8_lossy(&init.stderr)
        );

        let first_output = Command::new(env!("CARGO_BIN_EXE_mtxdb"))
            .args(["import", "--dir"])
            .arg(&root)
            .arg(&first)
            .output()
            .expect("run mtxdb import first file");
        let first_stderr = String::from_utf8_lossy(&first_output.stderr);
        assert!(first_output.status.success(), "stderr: {first_stderr}");
        assert!(
            first_stderr.contains("fail the state-before-event auth check")
                && first_stderr.contains("$bad"),
            "the first file should report its disagreement: {first_stderr}"
        );

        let second_output = Command::new(env!("CARGO_BIN_EXE_mtxdb"))
            .args(["import", "--dir"])
            .arg(&root)
            .arg(&second)
            .output()
            .expect("run mtxdb import second file");
        let second_stderr = String::from_utf8_lossy(&second_output.stderr);
        assert!(second_output.status.success(), "stderr: {second_stderr}");
        assert!(
            !second_stderr.contains("$bad"),
            "the second file must not report the earlier file's disagreement: {second_stderr}"
        );

        let _ = std::fs::remove_dir_all(root);
    }
}
