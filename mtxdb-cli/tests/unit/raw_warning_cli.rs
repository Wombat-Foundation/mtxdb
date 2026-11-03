//! Raw-import warning output across multiple files.

#[cfg(test)]
mod tests {
    use std::process::Command;

    #[test]
    fn raw_import_warns_on_conflicting_auth_projection() {
        let root = std::env::temp_dir().join(format!(
            "mtxdb-raw-projection-cli-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let first = root.join("first.jsonl");
        std::fs::write(
            &first,
            concat!(
                r#"{"event_id":"$create","sender":"@alice:test","room_id":"!room:test","type":"m.room.create","state_key":"","content":{"room_version":"10"},"prev_events":[],"auth_events":[]}"#,
                "\n",
                r#"{"event_id":"$join","sender":"@alice:test","room_id":"!room:test","type":"m.room.member","state_key":"@alice:test","content":{"membership":"join"},"prev_events":["$create"],"auth_events":["$create"]}"#,
                "\n",
                r#"{"event_id":"$pl","sender":"@alice:test","room_id":"!room:test","type":"m.room.power_levels","state_key":"","content":{"users":{"@alice:test":100},"invite":50},"prev_events":["$join"],"auth_events":["$create","$join"]}"#,
                "\n",
            ),
        )
        .unwrap();
        let second = root.join("second.jsonl");
        std::fs::write(
            &second,
            concat!(
                r#"{"event_id":"$pl","sender":"@alice:test","room_id":"!room:test","type":"m.room.power_levels","state_key":"","content":{"users":{"@alice:test":100},"invite":0},"prev_events":["$join"],"auth_events":["$create","$join"]}"#,
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
        assert!(second_output.status.success(), "stderr: {stderr}");
        assert!(
            stderr.contains("  warning: auth projection for event $pl conflicts with the stored projection; keeping the first"),
            "the second import must report the conflicting projection: {stderr}"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn raw_import_warnings_only_name_current_file_events() {
        let root = std::env::temp_dir().join(format!(
            "mtxdb-raw-warning-cli-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
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

        let output = Command::new(env!("CARGO_BIN_EXE_mtxdb"))
            .args(["import", "--dir"])
            .arg(&root)
            .arg(&first)
            .arg(&second)
            .output()
            .expect("run mtxdb import");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "stderr: {stderr}");
        assert!(
            stderr.contains("fail the state-before-event auth check"),
            "the first file should report the disagreement: {stderr}"
        );
        assert_eq!(
            stderr.matches("$bad").count(),
            1,
            "the earlier file's disagreement must not be reported again: {stderr}"
        );

        let _ = std::fs::remove_dir_all(root);
    }
}
