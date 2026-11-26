use super::*;

#[test]
fn test_dir_parsing() {
    let m = build_cli()
        .try_get_matches_from(["mtxdb", "packs", "list", "-d", "dir1", "-d", "dir2"])
        .unwrap();
    let dirs: Vec<_> = m
        .get_many::<String>("dir")
        .unwrap()
        .map(String::as_str)
        .collect();
    assert_eq!(dirs, vec!["dir1", "dir2"]);

    let m = build_cli()
        .try_get_matches_from(["mtxdb", "packs", "list", "-d", "dir1", "-d", "dir2"])
        .unwrap();
    let dirs: Vec<_> = m
        .get_many::<String>("dir")
        .unwrap()
        .map(String::as_str)
        .collect();
    assert_eq!(dirs, vec!["dir1", "dir2"]);

    let m = build_cli()
        .try_get_matches_from(["mtxdb", "packs", "list", "-d", "dir1", "-d", "dir2", "-a"])
        .unwrap();
    let dirs: Vec<_> = m
        .get_many::<String>("dir")
        .unwrap()
        .map(String::as_str)
        .collect();
    assert_eq!(dirs, vec!["dir1", "dir2"]);

    let m = build_cli()
        .try_get_matches_from([
            "mtxdb",
            "scan",
            "0x0102030405060708090a0b0c0d0e0f10",
            "-d",
            "dir1",
            "-d",
            "dir2",
        ])
        .unwrap();
    let dirs: Vec<_> = m
        .get_many::<String>("dir")
        .unwrap()
        .map(String::as_str)
        .collect();
    assert_eq!(dirs, vec!["dir1", "dir2"]);
    assert_eq!(m.subcommand_name(), Some("scan"));

    let m = build_cli()
        .try_get_matches_from(["mtxdb", "packs", "list", "-d", "dir1", "-d", "dir2", "-c"])
        .unwrap();
    assert!(m
        .subcommand_matches("packs")
        .unwrap()
        .subcommand_matches("list")
        .unwrap()
        .get_flag("coalesce"));

    // `--coalesce` belongs to the commands that aggregate; a leading flag or a
    // command that does not aggregate rejects it.
    for argv in [
        vec!["mtxdb", "--coalesce", "packs", "list"],
        vec!["mtxdb", "room", "timeline", "!r:x", "--coalesce"],
        vec!["mtxdb", "room", "members", "!r:x", "-c"],
        vec!["mtxdb", "put", "-c"],
    ] {
        assert!(build_cli().try_get_matches_from(argv).is_err());
    }
    // Nested `packs` subcommands inherit it from `packs`.
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "packs", "list", "-c"])
        .is_ok());

    let m = build_cli()
        .try_get_matches_from([
            "mtxdb", "repack", "-d", "dir1", "-d", "dir2", "-c", "--out", "target", "-y",
        ])
        .unwrap();
    let sub = m.subcommand_matches("repack").unwrap();
    assert!(sub.get_flag("coalesce"));
    assert_eq!(
        sub.get_one::<String>("out").map(String::as_str),
        Some("target")
    );
    assert!(sub.get_flag("yes"));
}

#[test]
fn info_is_collection_scoped_and_tombstones_need_a_selector() {
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "info", "0x1f"])
        .is_ok());
    assert!(build_cli()
        .try_get_matches_from([
            "mtxdb",
            "info",
            "!room:server",
            "--tombstones",
            "--keys",
            "--json"
        ])
        .is_ok());
    assert!(build_cli().try_get_matches_from(["mtxdb", "info"]).is_ok());
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "info", "--tombstones"])
        .is_err());
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "info", "!room:server", "--keys"])
        .is_err());
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "info", "--pack", "0x1f"])
        .is_err());
}

#[test]
fn optional_decode_formats_require_equals_and_default_to_auto() {
    fn check(bare: &[&str], explicit: &[&str]) {
        let bare_matches = build_cli().try_get_matches_from(bare).unwrap();
        let subcommand = bare_matches.subcommand_matches(bare[1]).unwrap();
        assert_eq!(
            subcommand.get_one::<String>("decode").map(String::as_str),
            Some("auto")
        );

        let explicit_matches = build_cli().try_get_matches_from(explicit).unwrap();
        let subcommand = explicit_matches.subcommand_matches(explicit[1]).unwrap();
        assert_eq!(
            subcommand.get_one::<String>("decode").map(String::as_str),
            Some("json")
        );

        let spaced = [bare[0], bare[1], bare[2], "--decode", "json"];
        assert!(build_cli().try_get_matches_from(spaced).is_err());
    }

    check(
        &["mtxdb", "meta", "wal", "--decode"],
        &["mtxdb", "meta", "wal", "--decode=json"],
    );
    check(
        &["mtxdb", "scan", "0x01", "--decode"],
        &["mtxdb", "scan", "0x01", "--decode=json"],
    );
    check(
        &["mtxdb", "get", "0x01", "--decode"],
        &["mtxdb", "get", "0x01", "--decode=json"],
    );
}

#[test]
fn test_read_plan_flag_parsing() {
    let read_plan = |argv: &[&str]| {
        let m = build_cli().try_get_matches_from(argv).unwrap();
        let mut leaf = &m;
        while let Some((_, sub)) = leaf.subcommand() {
            leaf = sub;
        }
        leaf.get_one::<String>("read_plan").cloned()
    };

    // Default is plain.
    assert_eq!(
        read_plan(&["mtxdb", "export", "0x01"]).as_deref(),
        Some("plain")
    );
    // Explicit prefetch is accepted.
    assert_eq!(
        read_plan(&["mtxdb", "export", "0x01", "--read-plan", "prefetch"]).as_deref(),
        Some("prefetch")
    );
    // An unknown mode is rejected by clap's value parser.
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "export", "0x01", "--read-plan", "ssd"])
        .is_err());
    // Commands that do not batch-read do not take it, nor does the root.
    for argv in [
        vec!["mtxdb", "packs", "list", "--read-plan", "prefetch"],
        vec!["mtxdb", "--read-plan", "prefetch", "export"],
        vec!["mtxdb", "init", "--read-plan", "prefetch"],
    ] {
        assert!(build_cli().try_get_matches_from(argv).is_err());
    }
}

#[test]
fn shard_type_is_scoped_to_pool_selecting_commands() {
    for argv in [
        ["mtxdb", "collections", "-t", "state"],
        ["mtxdb", "scan", "-t", "state"],
    ] {
        // `scan` needs a selector, so only the flag's acceptance matters.
        let result = build_cli().try_get_matches_from(argv);
        assert!(!matches!(
            result.as_ref().err().map(clap::Error::kind),
            Some(clap::error::ErrorKind::UnknownArgument)
        ));
    }
    for argv in [
        ["mtxdb", "init", "-t", "state"],
        ["mtxdb", "reclaim-wal", "-t", "state"],
        ["mtxdb", "meta", "-t", "state"],
        ["mtxdb", "-t", "state", "collections"],
    ] {
        assert!(build_cli().try_get_matches_from(argv).is_err());
    }
}

#[test]
fn scan_rejects_unknown_decode_format() {
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "scan", "0x01", "--decode=hff"])
        .is_err());
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "scan", "0x01", "--decode=hamt"])
        .is_ok());
}

#[test]
fn test_read_plan_mode_maps_to_policy() {
    assert_eq!(read_plan_from_mode("plain"), ReadPlanPolicy::disabled());
    assert_eq!(read_plan_from_mode("prefetch"), ReadPlanPolicy::prefetch());
    assert_ne!(read_plan_from_mode("prefetch"), ReadPlanPolicy::disabled());
}

#[test]
fn read_plan_modes_keep_the_runtime_point_mapping_setting() {
    for mode in ["plain", "prefetch"] {
        for enabled in [false, true] {
            let config = mtxdb::RuntimeConfig {
                point_random_mmap: enabled,
                ordered_index_point_random_mmap: enabled,
                ..mtxdb::RuntimeConfig::default()
            };
            let mut plan = read_plan_from_mode(mode);
            apply_runtime_read_settings(&mut plan, &config);
            assert_eq!(plan.point_random_mmap, enabled, "{mode}");
            assert_eq!(plan.point_random_mmap_index, enabled, "{mode}");
        }
    }
}

fn is_subcommand(name: &str) -> bool {
    build_cli().find_subcommand(name).is_some()
}

#[test]
fn dir_expansion_keeps_subcommand_file_operands() {
    let args = ["mtxdb", "import", "-d", "db", "events.json"];
    let expanded = expand_dir_args(
        args.map(Into::into),
        |p| p.to_string_lossy() == "db",
        |p| p.to_string_lossy().ends_with(".json"),
        is_subcommand,
    );
    assert_eq!(expanded, args.map(std::ffi::OsString::from));
}

#[test]
fn dir_expansion_does_not_rewrite_a_directory_named_like_a_subcommand() {
    let args = ["mtxdb", "-d", "pid-1", "collections"];
    let expanded = expand_dir_args(
        args.map(Into::into),
        |p| matches!(p.to_string_lossy().as_ref(), "pid-1" | "collections"),
        |_| false,
        is_subcommand,
    );
    assert_eq!(expanded, args.map(std::ffi::OsString::from));
}

#[test]
fn dir_flag_takes_a_shell_glob_of_directories() {
    let is_dir = |p: &Path| p.to_string_lossy().starts_with("pid-");
    let args = [
        "mtxdb",
        "-d",
        "pid-1",
        "pid-2",
        "collections",
        "-c",
        "-t",
        "all",
    ];
    let expanded = expand_dir_args(args.map(Into::into), is_dir, |_| false, is_subcommand);
    let m = build_cli().try_get_matches_from(expanded).unwrap();
    let dirs: Vec<_> = m
        .get_many::<String>("dir")
        .unwrap()
        .map(String::as_str)
        .collect();
    assert_eq!(dirs, vec!["pid-1", "pid-2"]);
    assert_eq!(m.subcommand_name(), Some("collections"));
}

#[test]
fn dir_expansion_leaves_non_directory_tokens_alone() {
    let args = ["mtxdb", "get", "-d", "pid-1", "some-key"];
    let expanded = expand_dir_args(
        args.map(Into::into),
        |p| p.to_string_lossy() == "pid-1",
        |_| false,
        is_subcommand,
    );
    assert_eq!(expanded, args.map(std::ffi::OsString::from));
}

#[test]
fn dir_expansion_drops_stray_files_from_a_glob() {
    let args = [
        "mtxdb",
        "-d",
        "pid-1",
        "pid-2",
        "pids.tar",
        "collections",
        "-c",
    ];
    let expanded = expand_dir_args(
        args.map(Into::into),
        |p| p.to_string_lossy().starts_with("pid-"),
        |p| p.to_string_lossy().ends_with(".tar"),
        is_subcommand,
    );
    let m = build_cli().try_get_matches_from(expanded).unwrap();
    let dirs: Vec<_> = m
        .get_many::<String>("dir")
        .unwrap()
        .map(String::as_str)
        .collect();
    assert_eq!(dirs, vec!["pid-1", "pid-2"]);
    assert_eq!(m.subcommand_name(), Some("collections"));
}

#[test]
fn test_shard_type_flag_accepts_server_info() {
    for val in ["server-info", "server_info"] {
        let m = build_cli()
            .try_get_matches_from(["mtxdb", "collections", "-t", val])
            .unwrap();
        assert_eq!(
            m.subcommand_matches("collections")
                .unwrap()
                .get_one::<String>("shard_type")
                .map(String::as_str),
            Some(val)
        );
        assert_eq!(shard_type_from_arg(val), Some(ShardType::ServerInfo));
    }
}

#[test]
fn search_requires_a_term_and_defaults_its_limit() {
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "search", "0x01"])
        .is_err());
    let matches = build_cli()
        .try_get_matches_from([
            "mtxdb",
            "search",
            "0x01",
            "--term",
            "hi",
            "--template",
            "t.json",
        ])
        .unwrap();
    let sub = matches.subcommand_matches("search").unwrap();
    assert_eq!(
        sub.get_one::<String>("term").map(String::as_str),
        Some("hi")
    );
    assert_eq!(
        sub.get_one::<String>("template").map(String::as_str),
        Some("t.json")
    );
    assert_eq!(sub.get_one::<i64>("limit"), Some(&50));
}

#[test]
fn gaps_anchor_requires_an_explicit_boundary() {
    let base = [
        "mtxdb",
        "room",
        "gaps",
        "!r:t",
        "anchor",
        "--gap",
        "$g",
        "--event",
        "$e",
        "--source",
        "state_ids",
    ];
    // `/state_ids` is the state before an event, so a missing boundary must be
    // an error rather than a guessed default.
    assert!(build_cli().try_get_matches_from(base).is_err());
    for boundary in ["before", "after"] {
        let mut args = base.to_vec();
        args.extend(["--boundary", boundary]);
        assert!(build_cli().try_get_matches_from(args).is_ok(), "{boundary}");
    }
    let mut bad = base.to_vec();
    bad.extend(["--boundary", "during"]);
    assert!(build_cli().try_get_matches_from(bad).is_err());
}

#[test]
fn room_check_defaults_to_both_audits_and_can_select_one() {
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "room", "check", "!r:t"])
        .is_ok());
    assert!(build_cli()
        .try_get_matches_from([
            "mtxdb",
            "room",
            "check",
            "!r:t",
            "--auth",
            "--state-members"
        ])
        .is_ok());
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "room", "check", "!r:t", "--prevs", "--history"])
        .is_ok());
    assert!(build_cli()
        .try_get_matches_from([
            "mtxdb",
            "room",
            "check",
            "!r:t",
            "--state-members",
            "--prevs"
        ])
        .is_ok());
    // A status filter is accepted with the default (both) selection.
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "room", "check", "!r:t", "--status", "open"])
        .is_ok());
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "room", "check", "!r:t", "--status", "closed"])
        .is_err());
}

#[test]
fn gaps_waive_needs_who_and_why() {
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "room", "gaps", "!r:t", "waive", "--gap", "$g"])
        .is_err());
}

#[test]
fn allow_waived_gaps_needs_the_strict_flag() {
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "import", "--allow-waived-gaps", "x.jsonl"])
        .is_err());
    assert!(build_cli()
        .try_get_matches_from([
            "mtxdb",
            "import",
            "--require-complete-prevs",
            "--allow-waived-gaps",
            "x.jsonl"
        ])
        .is_ok());
}

/// Every path in a scope table must name a real command that ends up with the
/// flag; a stale name would otherwise only surface as a clap panic at startup.
#[test]
fn scope_tables_name_real_commands_that_carry_their_flag() {
    let root = build_cli();
    for (flag, table) in [
        ("coalesce", COALESCE_COMMANDS),
        ("read_plan", READ_PLAN_COMMANDS),
        ("shard_type", SHARD_TYPE_COMMANDS),
    ] {
        for path in table {
            let mut command = &root;
            for name in *path {
                command = command
                    .find_subcommand(name)
                    .unwrap_or_else(|| panic!("`{}` is not a command", path.join(" ")));
            }
            assert!(
                command.get_arguments().any(|arg| arg.get_id() == flag),
                "`{}` is in the {flag} table but does not carry it",
                path.join(" ")
            );
        }
    }
}

/// Every synopsis in the README's CLI-layout block must parse against the real
/// builder. Placeholders become syntactically valid values and bracketed groups
/// are treated as present, so this catches drift such as a flag that became a
/// positional or a flag that was renamed.
#[test]
fn readme_cli_layout_synopses_parse() {
    const HEX32: &str = "00112233445566778899aabbccddeeff";
    let readme = include_str!("../../README.md");
    let mut in_section = false;
    let mut in_block = false;
    let mut checked = 0;

    for raw in readme.lines() {
        let line = raw.trim();
        if line.starts_with("## ") {
            in_section = line == "## CLI layout";
            in_block = false;
            continue;
        }
        if !in_section {
            continue;
        }
        if line.starts_with("```") {
            in_block = !in_block;
            continue;
        }
        if !in_block || !line.starts_with("mtxdb ") {
            continue;
        }

        let expanded = line
            .replace(
                "{ingest|waive|anchor}",
                "ingest --run run-1 /tmp/report.json",
            )
            .replace("<PACK_ID>", &format!("0x{HEX32}"))
            .replace("<COLLECTION>", &format!("0x{HEX32}"))
            .replace("<ROOM_ID>", "!room:example.org")
            .replace("<SELECTOR>", "!room:example.org")
            .replace("<STATUS>", "open")
            .replace("<FILE>", "/tmp/out.pack")
            .replace("<ID>", &format!("0x{HEX32}"))
            .replace(['[', ']'], "");
        let argv: Vec<&str> = expanded.split_whitespace().collect();
        assert_eq!(argv[0], "mtxdb");

        let result = build_cli().try_get_matches_from(&argv);
        assert!(
            result.is_ok(),
            "README synopsis does not parse: `{line}`\n{}",
            result.unwrap_err()
        );
        checked += 1;
    }

    assert!(
        checked >= 10,
        "README CLI layout block not found or too small"
    );
}
