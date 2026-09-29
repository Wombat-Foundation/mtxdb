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
fn test_read_plan_mode_maps_to_policy() {
    assert_eq!(read_plan_from_mode("plain"), ReadPlanPolicy::disabled());
    assert_eq!(read_plan_from_mode("prefetch"), ReadPlanPolicy::prefetch());
    assert_ne!(read_plan_from_mode("prefetch"), ReadPlanPolicy::disabled());
}
