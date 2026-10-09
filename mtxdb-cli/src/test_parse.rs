use super::*;

#[test]
fn test_dir_parsing() {
    let m = build_cli()
        .try_get_matches_from(["mtxdb", "shards", "-d", "dir1", "-d", "dir2"])
        .unwrap();
    let dirs: Vec<_> = m
        .get_many::<String>("dir")
        .unwrap()
        .map(String::as_str)
        .collect();
    assert_eq!(dirs, vec!["dir1", "dir2"]);

    let m = build_cli()
        .try_get_matches_from(["mtxdb", "shards", "-d", "dir1", "-d", "dir2"])
        .unwrap();
    let dirs: Vec<_> = m
        .get_many::<String>("dir")
        .unwrap()
        .map(String::as_str)
        .collect();
    assert_eq!(dirs, vec!["dir1", "dir2"]);

    let m = build_cli()
        .try_get_matches_from(["mtxdb", "shards", "-d", "dir1", "-d", "dir2", "-a"])
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
        .try_get_matches_from(["mtxdb", "shards", "-d", "dir1", "-d", "dir2", "-c"])
        .unwrap();
    assert!(m.get_flag("coalesce"));

    let m = build_cli()
        .try_get_matches_from(["mtxdb", "--coalesce", "shards", "-d", "dir1", "-d", "dir2"])
        .unwrap();
    assert!(m.get_flag("coalesce"));

    let m = build_cli()
        .try_get_matches_from([
            "mtxdb", "repack", "-d", "dir1", "-d", "dir2", "-c", "--out", "target", "-y",
        ])
        .unwrap();
    assert!(m.get_flag("coalesce"));
    let sub = m.subcommand_matches("repack").unwrap();
    assert_eq!(
        sub.get_one::<String>("out").map(String::as_str),
        Some("target")
    );
    assert!(sub.get_flag("yes"));
}

#[test]
fn test_info_selector_modes_and_mutual_exclusion() {
    // Positional, --pack, and --collection are each accepted alone.
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "info", "0x1f"])
        .is_ok());
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "info", "--pack", "0x1f"])
        .is_ok());
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "info", "--collection", "!room:server"])
        .is_ok());
    assert!(build_cli().try_get_matches_from(["mtxdb", "info"]).is_ok());

    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "info", "--pack", "0x1f", "--collection", "0x2f"])
        .is_err());
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "info", "0x1f", "--pack", "0x2f"])
        .is_err());
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "info", "0x1f", "--collection", "0x2f"])
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
    // Default is plain.
    let default = build_cli()
        .try_get_matches_from(["mtxdb", "shards"])
        .unwrap();
    assert_eq!(
        default.get_one::<String>("read_plan").map(String::as_str),
        Some("plain")
    );

    // Explicit prefetch is accepted (before and after the subcommand).
    for argv in [
        ["mtxdb", "--read-plan", "prefetch", "shards"],
        ["mtxdb", "shards", "--read-plan", "prefetch"],
    ] {
        let m = build_cli().try_get_matches_from(argv).unwrap();
        assert_eq!(
            m.get_one::<String>("read_plan").map(String::as_str),
            Some("prefetch")
        );
    }

    // An unknown mode is rejected by clap's value parser.
    assert!(build_cli()
        .try_get_matches_from(["mtxdb", "--read-plan", "ssd", "shards"])
        .is_err());
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
fn dir_flag_takes_a_shell_glob_of_directories() {
    let is_dir = |p: &Path| p.to_string_lossy().starts_with("pid-");
    let args = ["mtxdb", "-c", "-d", "pid-1", "pid-2", "collections", "-t", "all"];
    let expanded = expand_dir_args(args.map(Into::into), is_dir, |_| false);
    let m = build_cli().try_get_matches_from(expanded).unwrap();
    let dirs: Vec<_> = m.get_many::<String>("dir").unwrap().map(String::as_str).collect();
    assert_eq!(dirs, vec!["pid-1", "pid-2"]);
    assert_eq!(m.subcommand_name(), Some("collections"));
}

#[test]
fn dir_expansion_leaves_non_directory_tokens_alone() {
    let args = ["mtxdb", "get", "-d", "pid-1", "some-key"];
    let expanded = expand_dir_args(args.map(Into::into), |p| p.to_string_lossy() == "pid-1", |_| false);
    assert_eq!(expanded, args.map(std::ffi::OsString::from));
}

#[test]
fn dir_expansion_drops_stray_files_from_a_glob() {
    let args = ["mtxdb", "-c", "-d", "pid-1", "pid-2", "pids.tar", "collections"];
    let expanded = expand_dir_args(
        args.map(Into::into),
        |p| p.to_string_lossy().starts_with("pid-"),
        |p| p.to_string_lossy().ends_with(".tar"),
    );
    let m = build_cli().try_get_matches_from(expanded).unwrap();
    let dirs: Vec<_> = m.get_many::<String>("dir").unwrap().map(String::as_str).collect();
    assert_eq!(dirs, vec!["pid-1", "pid-2"]);
    assert_eq!(m.subcommand_name(), Some("collections"));
}
