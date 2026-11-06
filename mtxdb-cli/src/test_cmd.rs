use super::{
    base_is_incomplete, build_event_dag, canonical_column_width, cmd_collections, cmd_flags,
    cmd_get, cmd_import, cmd_import_file, cmd_info, cmd_remove, cmd_repack_coalesced, cmd_scan,
    cmd_scan_collection_live_rows, cmd_shards, cmd_stats, cmd_sync, cmd_tombstones,
    collection_canonical_id, compile_import_template, compute_state_groups_partial,
    decode_event_json_record, decode_hamt_node, decode_hamt_root, default_matrix_import_template,
    derive_template_key, display_collection_role, event_id, event_room_id, event_short_id,
    export_envelope_line, extract_pointer_string, finish_import, fmt_disk_megabytes, fmt_megabytes,
    format_canonical_display, format_id, glob_pack_files, import_data, import_pdu_events,
    import_pdu_events_with_raw, interleaving_worth_noting, is_server_keys_export, keys_first,
    listing_shard_types, load_collection_events, load_state_groups, matrix_batch_has_create,
    matrix_room_collection_id, matrix_room_extension_from_store, meta_checkpoints, meta_lock_line,
    meta_pools, meta_raw, overlay_auth_projection, pack_identity, parse_federation_input,
    parse_pack_id_selector, parse_pack_selectors, prepare_state_collection, pretty_print_payload,
    raw_auth_projection, raw_auth_projection_collection_id, record_csr_adjacency,
    redacted_event_bytes, resolve_import_collection, run, scan_payload_suffix,
    search_record_matches, search_template, split_canonical_display, template_collection_id,
    template_event_id_node_id, template_node_id, topological_event_order, valid_json_pointer,
    valid_state_group_id, verify_auth_chain_edges, write_raw_auth_projections, CollectionFlags,
    CollectionOptions, CollectionTemplate, ImportRun, JsonValueExt, MatrixRoomExtension,
    MetaReport, PackIdentity, ScanOptions, SharedState, StateGroupLoad, StateSet,
    MATRIX_ROOM_MEMBER_NAMESPACE, STATE_GROUP_ID_LENGTH, STATE_GROUP_NAMESPACE,
};
use crate::{export_import, Cli, Commands};
use bytes::Bytes;
use mtxdb::packfile::storage::PackfileStorage;
use mtxdb::packfile::PackId;
use mtxdb::room_auth::RoomAuth;
use mtxdb::shard::ShardPool;
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::template::{
    CollectionKeyRule, CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
};
use mtxdb::{
    content_digest, derive_collection_id, Database, DatabaseLayout, DigestAlgorithm,
    MatrixRoomVersion, ShardType,
};
use rezzy::json::Value as OwnedValue;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use base64::Engine;

fn owned_value(json: &str) -> OwnedValue {
    let bytes = json.as_bytes().to_vec();
    OwnedValue::parse_bytes(&bytes).expect("valid JSON fixture")
}

fn raw_event(
    room: &str,
    id: &str,
    kind: &str,
    state_key: &str,
    content: &str,
    prev: &str,
    auth: &str,
) -> OwnedValue {
    owned_value(&format!(
        r#"{{"event_id":"{id}","sender":"@alice:test","room_id":"{room}","type":"{kind}","state_key":"{state_key}","content":{content},"prev_events":{prev},"auth_events":{auth}}}"#
    ))
}

fn with_content_hash(mut event: OwnedValue, room_version: &str) -> OwnedValue {
    let hash =
        rezzy::compute_content_hash(&event, room_version).expect("content hash is computable");
    if let OwnedValue::Object(fields) = &mut event {
        let mut hashes = rezzy::json::Object::new();
        hashes.insert("sha256".to_owned(), OwnedValue::String(hash));
        fields.insert("hashes".to_owned(), OwnedValue::Object(hashes));
    }
    event
}

/// A power-levels event whose content says `invite: 50` but whose hash belongs
/// to an honest `invite: 0` event. The hash must fail, so this content may not
/// seed an auth projection.
fn forged_power_levels_with_stale_hash(room: &str) -> OwnedValue {
    let mut event = raw_event(
        room,
        "$pl",
        "m.room.power_levels",
        "",
        r#"{"users":{"@alice:test":100},"invite":50}"#,
        r#"["$join_alice"]"#,
        r#"["$create","$join_alice"]"#,
    );
    let honest_hash = rezzy::compute_content_hash(
        &raw_event(
            room,
            "$pl",
            "m.room.power_levels",
            "",
            r#"{"users":{"@alice:test":100},"invite":0}"#,
            r#"["$join_alice"]"#,
            r#"["$create","$join_alice"]"#,
        ),
        "10",
    )
    .expect("content hash is computable");
    if let OwnedValue::Object(fields) = &mut event {
        let mut hashes = rezzy::json::Object::new();
        hashes.insert("sha256".to_owned(), OwnedValue::String(honest_hash));
        fields.insert("hashes".to_owned(), OwnedValue::Object(hashes));
    }
    event
}

fn raw_v10_invite_fixture() -> [OwnedValue; 6] {
    [
        raw_event(
            "!room5",
            "$create",
            "m.room.create",
            "",
            r#"{"room_version":"10"}"#,
            "[]",
            "[]",
        ),
        raw_event(
            "!room5",
            "$join_alice",
            "m.room.member",
            "@alice:test",
            r#"{"membership":"join"}"#,
            r#"["$create"]"#,
            r#"["$create"]"#,
        ),
        // The stored form redacts `invite`, so its content hash will not match
        // on the next file. The projection must survive that check.
        with_content_hash(
            raw_event(
                "!room5",
                "$pl",
                "m.room.power_levels",
                "",
                r#"{"users":{"@alice:test":100},"invite":50}"#,
                r#"["$join_alice"]"#,
                r#"["$create","$join_alice"]"#,
            ),
            "10",
        ),
        raw_event(
            "!room5",
            "$rules",
            "m.room.join_rules",
            "",
            r#"{"join_rule":"public"}"#,
            r#"["$pl"]"#,
            r#"["$create","$join_alice","$pl"]"#,
        ),
        owned_value(
            r#"{"event_id":"$join_bob","sender":"@bob:test","room_id":"!room5","type":"m.room.member","state_key":"@bob:test","content":{"membership":"join"},"prev_events":["$rules"],"auth_events":["$create","$pl","$rules"]}"#,
        ),
        // Bob is level 0 and the room's invite level is 50.
        owned_value(
            r#"{"event_id":"$invite","sender":"@bob:test","room_id":"!room5","type":"m.room.member","state_key":"@carol:test","content":{"membership":"invite"},"prev_events":["$join_bob"],"auth_events":["$create","$pl","$join_bob"]}"#,
        ),
    ]
}

/// A string field of a JSON object, or `None` if the key is absent or not
/// a string. Used to assert on parsed output rather than substrings.
fn json_str(value: &OwnedValue, key: &str) -> Option<String> {
    match value.get(key) {
        Some(OwnedValue::String(text)) => Some(text.clone()),
        _ => None,
    }
}

#[test]
fn scan_renders_auxiliary_envelope_value() {
    let mut payload = b"AUX1".to_vec();
    payload.extend_from_slice(&[0xabu8; 32]);
    payload.extend_from_slice(b"state-group-id");
    assert_eq!(
        scan_payload_suffix(&payload),
        Some("AUX1 value: state-group-id".to_owned())
    );
}

#[test]
fn scan_reports_truncated_auxiliary_envelope() {
    assert_eq!(
        scan_payload_suffix(b"AUX1"),
        Some("4 bytes (malformed AUX1 envelope)".to_owned())
    );
}

#[test]
fn scan_renders_the_event_to_state_group_pointer() {
    let mut payload = b"AUX1".to_vec();
    payload.extend_from_slice(&[0xabu8; 32]);
    payload.extend_from_slice(&[0x12u8; STATE_GROUP_ID_LENGTH]);
    assert_eq!(
        scan_payload_suffix(&payload),
        Some(format!("PTR: 0x{}", "12".repeat(16)))
    );
}

#[test]
fn scan_decodes_a_state_group_instance() {
    let instance = mtxdb::StateGroupInstance {
        parents: vec![[0x11; 16], [0x22; 16]],
        lthash: [0x33; 32],
        root_id: [0x44; 16],
        derivation: mtxdb::StateGroupDerivation::new(
            mtxdb::StateGroupResolver::V2,
            mtxdb::StateGroupTrust::TrustedImport,
        ),
    };
    let payload = mtxdb::encode_state_group_record(&instance);
    // The one-line cell stays empty so `--verbose`/`--decode` prints the
    // recognized multi-line form instead.
    assert!(scan_payload_suffix(&payload).is_none());
    let decoded = super::decode_state_group(&payload).expect("STGP decode");
    let text = String::from_utf8(decoded).unwrap();
    assert!(text.contains("STGP state-group instance"));
    assert!(text.contains(&"11".repeat(16)));
    assert!(text.contains(&"33".repeat(32)));
    assert!(text.contains(&"44".repeat(16)));
    assert!(pretty_print_payload(&payload).is_some());
}

#[test]
fn immutable_record_filter_skips_duplicates_and_rejects_collisions() {
    use mtxdb::storage::InMemoryStorage;
    let store = InMemoryStorage::new();
    let collection = [0u8; 16];
    let id = [1u8; 16];

    let fresh =
        super::filter_new_records(&store, &collection, vec![(id, NodeData::from_slice(b"a"))])
            .unwrap();
    assert_eq!(fresh.len(), 1);
    store
        .put_many(&collection, &[(id, NodeData::from_slice(b"a"))])
        .unwrap();

    let fresh =
        super::filter_new_records(&store, &collection, vec![(id, NodeData::from_slice(b"a"))])
            .unwrap();
    assert!(fresh.is_empty(), "identical payload is skipped");

    assert!(
        super::filter_new_records(&store, &collection, vec![(id, NodeData::from_slice(b"b"))])
            .is_err(),
        "a differing payload under the same id is a hard collision"
    );
}

#[test]
fn canonical_collection_display_aligns_roles_and_reserves_total_column() {
    let short = "!short (event)".to_owned();
    let long = "!a-much-longer-room-id (event)".to_owned();
    assert_eq!(display_collection_role("event_dag"), "event");
    assert_eq!(display_collection_role("state"), "state");
    assert_eq!(split_canonical_display(&short), ("!short", Some("event)")));
    assert_eq!(
        canonical_column_width([&short, &long].into_iter()),
        "!a-much-longer-room-id".len()
    );
    assert_eq!(
        format_canonical_display(&short, "!a-much-longer-room-id".len(), "(event)".len(),),
        format!(
            "!short{}  (event)",
            " ".repeat("!a-much-longer-room-id".len() - "!short".len())
        )
    );
}

fn unique_temp_dir() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "mtxdb-cli-test-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn meta_json_summary_has_numeric_severity_counts() {
    let mut report = MetaReport::default();
    report.finding("NOTE", Path::new("note"), "transient");
    report.finding("WARN", Path::new("warn"), "mismatch");
    report.finding("ERROR", Path::new("error"), "unreadable");
    let value = report.json_value();
    let encoded = value.encode();
    let bytes = encoded.into_bytes();
    let parsed = OwnedValue::parse_bytes(&bytes).expect("meta JSON must parse");
    let OwnedValue::Array(records) = parsed else {
        panic!("meta JSON must be an array");
    };
    let OwnedValue::Object(summary) = &records[0] else {
        panic!("summary must be an object");
    };
    assert_eq!(
        summary.get("kind").and_then(OwnedValue::as_str),
        Some("summary")
    );
    assert_eq!(summary.get("note").and_then(OwnedValue::as_u64), Some(1));
    assert_eq!(summary.get("warn").and_then(OwnedValue::as_u64), Some(1));
    assert_eq!(summary.get("error").and_then(OwnedValue::as_u64), Some(1));
}

#[test]
fn meta_raw_paginates_across_files() {
    let root = unique_temp_dir();
    let state = root.join("pools/mtpl-state");
    std::fs::create_dir_all(&state).unwrap();
    std::fs::write(root.join("db.meta"), (0_u8..32).collect::<Vec<_>>()).unwrap();
    std::fs::write(state.join("pool.meta"), (32_u8..64).collect::<Vec<_>>()).unwrap();
    let mut report = MetaReport::default();
    meta_raw(&root, &mut report, 3, 1);
    let raw_lines = report
        .lines
        .iter()
        .filter(|line| line.contains('|'))
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(raw_lines.len(), 3);
    assert!(raw_lines[0].contains("00000010"));
    assert!(raw_lines[1].contains("00000000"));
    assert!(raw_lines[2].contains("00000010"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn meta_lock_reports_pid_confidence_without_probe() {
    let root = unique_temp_dir();
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join(".mtxdb.lock");
    std::fs::write(&path, format!("{}\n", std::process::id())).unwrap();
    let info = mtxdb::ShardPool::lock_holder_info(&path).expect("PID marker");
    assert_eq!(info.pid, std::process::id());
    assert!(info.running);
    assert!(!info.has_starttime);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn meta_lock_reports_full_confidence_for_starttime_marker() {
    let root = unique_temp_dir();
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join(".mtxdb.lock");
    std::fs::write(&path, b"1 1\n").unwrap();
    let mut report = MetaReport::default();
    meta_lock_line(&path, &mut report);
    assert!(report.lines[0].contains("confidence=full"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn noncanonical_pack_identity_is_a_note() {
    let root = unique_temp_dir();
    let state = root.join("pools/mtpl-state");
    std::fs::create_dir_all(&state).unwrap();
    let path = state.join("foreign.pack");
    std::fs::write(&path, b"not a canonical pack").unwrap();
    assert!(matches!(
        pack_identity(&path),
        PackIdentity::NonCanonical(_)
    ));
    let mut report = MetaReport::default();
    meta_pools(&root, &mut report, usize::MAX, 0, true);
    assert!(report.lines.iter().any(|line| line.starts_with("[NOTE]")));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn checkpoint_pack_comparison_excludes_noncanonical_pack_but_checks_valid_set() {
    let root = unique_temp_dir();
    let state = root.join("pools/mtpl-state");
    std::fs::create_dir_all(&state).unwrap();
    std::fs::write(state.join("foreign.pack"), b"not a canonical pack").unwrap();
    mtxdb::index::checkpoint::write_checkpoint(
        &state.join(mtxdb::index::checkpoint::INDEX_CHECKPOINT_FILE),
        0xdead_beef,
        0,
        0,
        &[],
        &[],
        &[],
    )
    .unwrap();
    let mut report = MetaReport::default();
    meta_checkpoints(&root, &mut report);
    assert!(report
        .lines
        .iter()
        .any(|line| line.contains("foreign.pack") && line.starts_with("[NOTE]")));
    assert!(report
        .lines
        .iter()
        .any(|line| line.contains("fingerprint mismatch") && line.starts_with("[WARN]")));
    assert!(!report
        .lines
        .iter()
        .any(|line| line.contains("fingerprint comparison skipped")));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn checkpoint_pack_comparison_is_quiet_when_packs_match() {
    let root = unique_temp_dir();
    let state = root.join("pools/mtpl-state");
    std::fs::create_dir_all(&state).unwrap();
    std::fs::write(state.join("foreign.pack"), b"not a canonical pack").unwrap();
    mtxdb::index::checkpoint::write_checkpoint(
        &state.join(mtxdb::index::checkpoint::INDEX_CHECKPOINT_FILE),
        mtxdb::index::checkpoint::pack_fingerprint(&[]),
        0,
        0,
        &[],
        &[],
        &[],
    )
    .unwrap();
    let mut report = MetaReport::default();
    meta_checkpoints(&root, &mut report);
    assert_eq!(report.warn_count, 0);
    assert_eq!(report.note_count, 1);
    assert!(report
        .lines
        .iter()
        .any(|line| line.contains("foreign.pack")));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn checkpoint_pack_comparison_skips_invalid_pack_with_explicit_note() {
    let root = unique_temp_dir();
    let state = root.join("pools/mtpl-state");
    std::fs::create_dir_all(&state).unwrap();
    std::fs::write(state.join("pack_0000000000000001.pack"), b"not a pack").unwrap();
    mtxdb::index::checkpoint::write_checkpoint(
        &state.join(mtxdb::index::checkpoint::INDEX_CHECKPOINT_FILE),
        mtxdb::index::checkpoint::pack_fingerprint(&[]),
        0,
        0,
        &[],
        &[],
        &[],
    )
    .unwrap();
    let mut report = MetaReport::default();
    meta_checkpoints(&root, &mut report);
    assert!(report
        .lines
        .iter()
        .any(|line| line.contains("fingerprint comparison skipped")));
    assert!(report
        .lines
        .iter()
        .any(|line| line.starts_with("[WARN]") && line.contains("pack_0000000000000001.pack")));
    assert!(!report
        .lines
        .iter()
        .any(|line| line.contains("fingerprint mismatch")));
    let _ = std::fs::remove_dir_all(root);
}

fn compile_reject(template: &[u8]) -> anyhow::Result<CollectionTemplate> {
    let dir = unique_temp_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("template.json");
    std::fs::write(&path, template).unwrap();
    let result = compile_import_template(Some(&path));
    let _ = std::fs::remove_dir_all(&dir);
    result
}

#[test]
fn interleaving_note_fires_only_on_average_fragmentation() {
    // The user's state-pool case: ~10 extra runs per collection.
    assert!(interleaving_worth_noting(10319, 104_933));
    // One extra run per collection on average is the threshold.
    assert!(interleaving_worth_noting(250, 250));
    // A few collections each split a couple of times is noise.
    assert!(!interleaving_worth_noting(250, 200));
    // No interleaving at all, or tiny stores, stays quiet.
    assert!(!interleaving_worth_noting(250, 0));
    assert!(!interleaving_worth_noting(1, 0));
    // Degenerate guards (unit-test the clamp, not realistic data).
    assert!(interleaving_worth_noting(0, 1));
    assert!(!interleaving_worth_noting(0, 0));
}

#[test]
fn index_memory_megabytes_uses_fixed_point_rounding() {
    assert_eq!(fmt_megabytes(524_328), "0.52433 MB");
    assert_eq!(fmt_megabytes(999_999), "1.00000 MB");
}

#[test]
fn disk_megabytes_uses_three_fractional_digits() {
    assert_eq!(fmt_disk_megabytes(42_280), "0.042 MB");
    assert_eq!(fmt_disk_megabytes(999_999), "1.000 MB");
}

#[test]
fn sync_all_does_not_materialize_empty_pools() {
    let dir = unique_temp_dir();
    let layout = DatabaseLayout::open(dir.clone()).unwrap();
    let cli = Cli {
        dirs: vec![dir.clone()],
        shard_type: Some(ShardType::State),
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Sync { all: true },
    };

    cmd_sync(&cli, true).unwrap();

    for shard_type in ShardType::ALL {
        // Pools are created by their first write, so a pool nothing wrote to has
        // no directory at all; either way it must hold no pack.
        let pool = layout.pool_path(shard_type);
        assert!(
            glob_pack_files(&pool).unwrap().is_empty(),
            "sync --all must not create a pack in the empty {} pool",
            shard_type.as_str()
        );
    }
    drop(layout);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn shard_types_yields_all_three_when_no_type_is_selected() {
    // `-t all` parses to `shard_type: None` (main.rs's `"all" => None`
    // match arm) -- this is the exact case that used to make
    // `require_shard_type` reject `collections`/`shards` even though
    // `-t all` completes and parses as a legitimate value.
    let cli = Cli {
        dirs: Vec::new(),
        shard_type: None,
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Collections {
            per_db: false,
            all: false,
            layout: false,
            canonical: false,
            sort: None,
            limit: -1,
        },
    };
    assert_eq!(
        cli.shard_types().collect::<Vec<_>>(),
        ShardType::ALL.to_vec()
    );
}

#[test]
fn shard_types_yields_just_the_selected_type() {
    let cli = Cli {
        dirs: Vec::new(),
        shard_type: Some(ShardType::State),
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Collections {
            per_db: false,
            all: false,
            layout: false,
            canonical: false,
            sort: None,
            limit: -1,
        },
    };
    assert_eq!(
        cli.shard_types().collect::<Vec<_>>(),
        vec![ShardType::State]
    );
}

#[test]
fn listing_all_overrides_the_default_pool_selection() {
    let cli = Cli {
        dirs: Vec::new(),
        shard_type: Some(ShardType::EventDag),
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Collections {
            per_db: false,
            all: true,
            layout: false,
            canonical: false,
            sort: None,
            limit: -1,
        },
    };
    assert_eq!(listing_shard_types(&cli, true), ShardType::ALL.to_vec());
}

#[test]
fn collections_dash_t_all_iterates_every_pool_without_the_all_flag() {
    // Regression test for the `-t all` vs `--all` inconsistency: before
    // the fix, `shard_type: None` with `all: false` (i.e. `-t all` typed
    // without also passing `--all`) errored with "this command requires
    // a specific shard type" instead of behaving like `--all`.
    let dir = unique_temp_dir();
    DatabaseLayout::open(dir.clone()).unwrap();
    let cli = Cli {
        dirs: vec![dir.clone()],
        shard_type: None,
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Collections {
            per_db: false,
            all: false,
            layout: false,
            canonical: false,
            sort: None,
            limit: -1,
        },
    };

    cmd_collections(
        &cli,
        &CollectionOptions {
            flags: CollectionFlags::from_bools([false, false, false, false]),
            sort: None,
            limit: -1,
        },
    )
    .unwrap();

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn shards_dash_t_all_iterates_every_pool_without_the_all_flag() {
    let dir = unique_temp_dir();
    DatabaseLayout::open(dir.clone()).unwrap();
    let cli = Cli {
        dirs: vec![dir.clone()],
        shard_type: None,
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Shards {
            all: false,
            layout: false,
            sort: None,
        },
    };

    cmd_shards(&cli, false, false, None).unwrap();

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn collections_with_a_specific_type_still_targets_only_that_pool() {
    // Unaffected-path guard: a specific `-t` selection (the default,
    // and the common case) must still behave exactly as before --
    // single-pool, no iteration, no `open_layout` overhead.
    let dir = unique_temp_dir();
    DatabaseLayout::open(dir.clone()).unwrap();
    let cli = Cli {
        dirs: vec![dir.clone()],
        shard_type: Some(ShardType::State),
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Collections {
            per_db: false,
            all: false,
            layout: false,
            canonical: false,
            sort: None,
            limit: -1,
        },
    };

    cmd_collections(
        &cli,
        &CollectionOptions {
            flags: CollectionFlags::from_bools([false, false, false, false]),
            sort: None,
            limit: -1,
        },
    )
    .unwrap();

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn stats_dash_t_all_iterates_every_pool() {
    let dir = unique_temp_dir();
    DatabaseLayout::open(dir.clone()).unwrap();
    let cli = Cli {
        dirs: vec![dir.clone()],
        shard_type: None,
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Stats { json: false },
    };
    cmd_stats(&cli, false).unwrap();
    cmd_stats(&cli, true).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn stats_json_quotes_the_pack_id() {
    // `PackId`'s Display is `0x<hex>`, which is not a bare JSON number, so
    // the shard entry must render it as a quoted string.
    let id = mtxdb::packfile::PackId::from_hex("ab12cd34ef567890ab12cd34ef567890").unwrap();
    let summaries = [mtxdb::shard::ShardSummary {
        slot: 0,
        pack_id: id,
        file_bytes: 4096,
        stats: mtxdb::shard::ShardStats::default(),
    }];
    let json = super::stats_json_object(
        std::path::Path::new("/nonexistent"),
        &mtxdb::packfile::storage::RuntimeStats::default(),
        &summaries,
    );
    assert!(
        json.contains("\"pack_id\":\"0xab12cd34ef567890ab12cd34ef567890\""),
        "{json}"
    );
}

#[test]
fn info_explains_a_malformed_hex_selector() {
    let cli = Cli {
        dirs: Vec::new(),
        shard_type: None,
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Info {
            selector: Some(String::new()),
            pack: None,
            collection: None,
            stats: false,
        },
    };
    let doubled = cmd_info(&cli, "0x0x144ACE34F53560B728FA9E33DD3FEF63", None, false).unwrap_err();
    assert!(doubled.to_string().contains("doubled"), "{doubled}");
    let short = cmd_info(&cli, "0x144ACE34F53560B728FA9E33DD3FEF", None, false).unwrap_err();
    assert!(short.to_string().contains("found 30 characters"), "{short}");
}

#[test]
fn info_selector_boundary_between_pack_and_collection() {
    let cli = Cli {
        dirs: Vec::new(),
        shard_type: None,
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Info {
            selector: None,
            pack: None,
            collection: None,
            stats: false,
        },
    };
    let digits = |n: usize| format!("0x{}", "a".repeat(n));
    assert_eq!(
        super::classify_info_selector(&cli, "7").unwrap(),
        super::InfoTarget::Collection
    );
    assert_eq!(
        super::classify_info_selector(&cli, &digits(1)).unwrap(),
        super::InfoTarget::Pack
    );
    assert_eq!(
        super::classify_info_selector(&cli, &digits(16)).unwrap(),
        super::InfoTarget::Pack
    );
    let upper = super::classify_info_selector(&cli, "0X1").unwrap_err();
    assert!(upper.to_string().contains("lowercase"), "{upper}");
    assert_eq!(
        super::classify_info_selector(&cli, &digits(32)).unwrap(),
        super::InfoTarget::Collection
    );
    for n in [0, 17, 31, 33, 64] {
        assert!(
            super::classify_info_selector(&cli, &digits(n)).is_err(),
            "{n} digits after 0x must be rejected"
        );
    }
}

#[test]
fn info_selector_32_hex_disambiguation() {
    let dir = unique_temp_dir();
    let layout = DatabaseLayout::open(dir.clone()).unwrap();
    let pool = layout.pool_dir(ShardType::State).unwrap();
    let store = mtxdb::PackfileStorage::open(pool.clone()).unwrap();

    // 1. Write a record to create a pack and collection
    let collection_a = [0x42u8; 16];
    let data = mtxdb::NodeData::new(bytes::Bytes::from_static(b"data1"));
    store.put(&collection_a, &[0x01; 16], &data).unwrap();
    store.sync_all().unwrap();

    let files = glob_pack_files(&pool).unwrap();
    let pack_id = files[0].0;
    let pack_hex = format!("0x{}", pack_id.as_hex());
    let col_hex = format_id(&collection_a);

    let cli = Cli {
        dirs: vec![dir.clone()],
        shard_type: Some(ShardType::State),
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Info {
            selector: None,
            pack: None,
            collection: None,
            stats: false,
        },
    };

    // Pack hex resolves to Pack
    assert_eq!(
        super::classify_info_selector(&cli, &pack_hex).unwrap(),
        super::InfoTarget::Pack
    );

    // Collection hex resolves to Collection
    assert_eq!(
        super::classify_info_selector(&cli, &col_hex).unwrap(),
        super::InfoTarget::Collection
    );

    // If a collection with the EXACT same ID as pack_id is created, it becomes ambiguous:
    let collision_data = mtxdb::NodeData::new(bytes::Bytes::from_static(b"collision"));
    store
        .put(pack_id.as_bytes(), &[0x02; 16], &collision_data)
        .unwrap();
    store.sync_all().unwrap();
    drop(store);

    // Positional inference now fails with actionable advice naming both
    // real flags (they exist and are verified below).
    let ambiguous_err = super::classify_info_selector(&cli, &pack_hex).unwrap_err();
    let msg = ambiguous_err.to_string();
    assert!(
        msg.contains("matches both a pack and a collection"),
        "{msg}"
    );
    assert!(msg.contains("--pack"), "{msg}");
    assert!(msg.contains("--collection"), "{msg}");

    // `--pack` bypasses inference and resolves the pack even under collision.
    assert_eq!(
        super::classify_info_selector_explicit(&cli, &pack_hex, Some(super::InfoTarget::Pack))
            .unwrap(),
        super::InfoTarget::Pack
    );
    // `--collection` bypasses inference and resolves the collection.
    assert_eq!(
        super::classify_info_selector_explicit(&cli, &col_hex, Some(super::InfoTarget::Collection))
            .unwrap(),
        super::InfoTarget::Collection
    );
    // A unique prefix resolves under `--pack`...
    let prefix = format!("0x{}", &pack_id.as_hex()[..16]);
    assert_eq!(
        super::classify_info_selector_explicit(&cli, &prefix, Some(super::InfoTarget::Pack))
            .unwrap(),
        super::InfoTarget::Pack
    );
    // ...and `--pack` with an unknown id errors rather than falling through.
    let missing = format!("0x{}", "ff".repeat(mtxdb::packfile::PACK_ID_LEN));
    assert!(
        super::classify_info_selector_explicit(&cli, &missing, Some(super::InfoTarget::Pack))
            .is_err()
    );
    // `--collection` never falls back to a pack: an address with no such
    // collection errors, even though it is a live pack.
    let no_collection = format!("0x{}", "ee".repeat(mtxdb::packfile::PACK_ID_LEN));
    assert!(super::classify_info_selector_explicit(
        &cli,
        &no_collection,
        Some(super::InfoTarget::Collection)
    )
    .is_err());
    // The deliberately-created colliding collection *does* resolve under
    // `--collection` for the pack's address — that is the point of the flag.
    assert_eq!(
        super::classify_info_selector_explicit(
            &cli,
            &pack_hex,
            Some(super::InfoTarget::Collection)
        )
        .unwrap(),
        super::InfoTarget::Collection
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn info_pack_flag_reports_same_id_in_multiple_pools() {
    // A full PackId present in two selected pools cannot be disambiguated
    // by using the full address — the error must say to narrow the pool.
    let dir1 = unique_temp_dir();
    let dir2 = unique_temp_dir();
    let l1 = DatabaseLayout::open(dir1.clone()).unwrap();
    let l2 = DatabaseLayout::open(dir2.clone()).unwrap();
    let p1 = l1.pool_dir(ShardType::EventDag).unwrap();
    let p2 = l2.pool_dir(ShardType::EventDag).unwrap();

    let s1 = PackfileStorage::open(p1.clone()).unwrap();
    s1.put(
        &[0x11; 16],
        &[0x33; 16],
        &NodeData::new(Bytes::from_static(b"a")),
    )
    .unwrap();
    // Fully durable before the file is copied into the second pool.
    s1.sync_all().unwrap();
    drop(s1);
    // Give dir2 its own pool, then copy dir1's pack into it so the same
    // PackId is live in both selected pools.
    let s2 = PackfileStorage::open(p2.clone()).unwrap();
    s2.put(
        &[0x22; 16],
        &[0x44; 16],
        &NodeData::new(Bytes::from_static(b"b")),
    )
    .unwrap();
    s2.sync_all().unwrap();
    drop(s2);

    let pack_file = std::fs::read_dir(&p1)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.extension().is_some_and(|ext| ext == "pack"))
        .expect("dir1 pool has a pack file");
    let pack_id = glob_pack_files(&p1).unwrap()[0].0;
    std::fs::copy(&pack_file, p2.join(pack_file.file_name().unwrap())).unwrap();

    let cli = Cli {
        dirs: vec![dir1.clone(), dir2.clone()],
        shard_type: Some(ShardType::EventDag),
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Info {
            selector: None,
            pack: None,
            collection: None,
            stats: false,
        },
    };

    let full = format!("0x{}", pack_id.as_hex());
    let err = super::classify_info_selector_explicit(&cli, &full, Some(super::InfoTarget::Pack))
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("selected pools"), "{msg}");
    assert!(msg.contains("--dir or -t"), "{msg}");

    std::fs::remove_dir_all(&dir1).ok();
    std::fs::remove_dir_all(&dir2).ok();
}

#[test]
fn only_the_canonical_lowercase_prefix_is_accepted() {
    let cli = Cli {
        dirs: Vec::new(),
        shard_type: None,
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Info {
            selector: None,
            pack: None,
            collection: None,
            stats: false,
        },
    };
    // A short 0x-prefixed prefix is a filename-prefix selector.
    let short = super::parse_pack_id_selector("0x1f").unwrap();
    assert!(short.matches(&PackId([0x1f; mtxdb::packfile::PACK_ID_LEN])));
    assert!(!short.matches(&PackId([0x20; mtxdb::packfile::PACK_ID_LEN])));
    // A full 32-hex address is an exact selector.
    let full = "ab".repeat(mtxdb::packfile::PACK_ID_LEN);
    assert!(super::parse_pack_id_selector(&format!("0x{full}")).is_ok());
    assert!(super::parse_pack_id_selector(&format!("0x{}", full.to_uppercase())).is_err());

    let id = format_id(&[0xABu8; 16]);
    let upper = id.replacen("0x", "0X", 1);
    assert!(super::parse_pack_id_selector("0X1f").is_err());
    assert!(super::parse_collection_id(&upper).is_err());
    assert!(super::parse_node_id(&upper).is_err());
    assert!(super::classify_info_selector(&cli, &upper).is_err());
    // Every selector `info` routes to a pack must parse as one.
    for selector in ["0x1", "0xffffffffffffffff"] {
        assert_eq!(
            super::classify_info_selector(&cli, selector).unwrap(),
            super::InfoTarget::Pack
        );
        assert!(
            super::parse_pack_id_selector(selector).is_ok(),
            "{selector}"
        );
    }
}

#[test]
fn collection_selectors_are_never_positional() {
    // A listing position names a different collection after a purge or
    // repack, so a bare number must not resolve to one.
    for selector in ["0", "3", "19"] {
        let error = super::parse_collection_selector(selector).unwrap_err();
        assert!(error.to_string().contains("0x-prefixed"), "{error}");
    }
}

#[test]
fn displayed_collection_ids_are_accepted_as_selectors() {
    let id = [0xABu8; 16];
    assert_eq!(super::parse_collection_id(&format_id(&id)).unwrap(), id);
    assert!(!format_id(&id).starts_with("0x0x"));
}

#[test]
fn export_envelope_carries_exact_bytes_and_frame_metadata() {
    let collection_id = [0x11u8; 16];
    let node_id = [0x22u8; 16];
    let metadata = mtxdb::packfile::FrameMetadata {
        logical_id: Some([0x33u8; 32]),
        content_digest: Some([0x44u8; 32]),
        digest_algorithm: DigestAlgorithm::Blake3,
        role: Some(b"event".to_vec()),
        last_write_lsn: None,
        removed: false,
        unknown: vec![(0x7f, vec![0xDE, 0xAD])],
    };
    // Deliberately non-canonical JSON (spaces) so a reserialized payload
    // would differ from the exact bytes.
    let payload = b"{ \"event_id\" : \"$x\" }";
    let line = export_envelope_line(&collection_id, &node_id, payload, Some(&metadata));

    let value = owned_value(&line);
    assert_eq!(
        json_str(&value, "schema").as_deref(),
        Some("mtxdb.export/v1")
    );
    assert_eq!(
        json_str(&value, "collection_id").as_deref(),
        Some("0x11111111111111111111111111111111")
    );
    assert_eq!(
        json_str(&value, "node_id").as_deref(),
        Some("0x22222222222222222222222222222222")
    );
    assert_eq!(
        json_str(&value, "payload_encoding").as_deref(),
        Some("json")
    );

    // `payload_base64` is the exact stored bytes, not the decoded value.
    let expected = base64::engine::general_purpose::STANDARD.encode(payload);
    assert_eq!(
        json_str(&value, "payload_base64").as_deref(),
        Some(expected.as_str())
    );
    let decoded = value.get("payload").expect("decoded JSON payload");
    assert_eq!(json_str(decoded, "event_id").as_deref(), Some("$x"));

    let meta = value.get("metadata").expect("metadata object");
    assert_eq!(
        json_str(meta, "digest_algorithm").as_deref(),
        Some("blake3")
    );
    assert_eq!(json_str(meta, "digest_algorithm_id").as_deref(), Some("2"));
    assert!(
        json_str(meta, "logical_id").unwrap().starts_with("0x3333"),
        "{line}"
    );
    let role = meta.get("role").expect("role object");
    assert_eq!(json_str(role, "encoding").as_deref(), Some("utf8"));
    assert_eq!(json_str(role, "value").as_deref(), Some("event"));
    let unknown = meta.get("unknown").expect("unknown array");
    assert_eq!(
        unknown.encode(),
        "[{\"tag\":127,\"value_base64\":\"3q0=\"}]"
    );
}

#[test]
fn export_envelope_base64_encodes_non_json_payloads() {
    let line = export_envelope_line(&[0x11u8; 16], &[0x22u8; 16], &[0xDE, 0xAD], None);
    let value = owned_value(&line);
    assert_eq!(
        json_str(&value, "payload_encoding").as_deref(),
        Some("base64")
    );
    assert_eq!(json_str(&value, "payload_base64").as_deref(), Some("3q0="));
    assert!(
        value.get("payload").is_none(),
        "binary payloads carry no decoded JSON value"
    );
    assert_eq!(value.get("metadata").unwrap().encode(), "null");
}

#[test]
fn packs_dump_writes_decoded_jsonl_for_a_pack() {
    let dir = unique_temp_dir();
    let layout = DatabaseLayout::open(dir.clone()).unwrap();
    let pool = layout.pool_dir(ShardType::EventDag).unwrap();
    let store = PackfileStorage::open(pool).unwrap();
    let collection_id = [0x11; 16];
    let node_id = [0x22; 16];
    let payload = b"{ \"event_id\" : \"$dump\" }";
    store
        .put(
            &collection_id,
            &node_id,
            &NodeData::new(Bytes::copy_from_slice(payload)),
        )
        .unwrap();
    store.sync().unwrap();
    let pack_id = store.shard_summaries()[0].pack_id;
    drop(store);

    let cli = Cli {
        dirs: vec![dir.clone()],
        shard_type: Some(ShardType::EventDag),
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Packs {
            action: super::PacksAction::List { all: false },
        },
    };
    let out_file = dir.join("dump.jsonl");
    super::cmd_packs_dump(&cli, &pack_id.to_string(), None, Some(&out_file)).unwrap();

    let content = std::fs::read_to_string(&out_file).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 1, "{content}");
    let value = owned_value(lines[0]);
    assert_eq!(
        json_str(&value, "schema").as_deref(),
        Some("mtxdb.pack.dump/v1")
    );
    assert_eq!(json_str(&value, "pool").as_deref(), Some("mtpl-event"));
    let expected_pack = pack_id.to_string();
    assert_eq!(
        json_str(&value, "pack_id").as_deref(),
        Some(expected_pack.as_str())
    );
    let expected_node = format!("0x{}", hex::encode(node_id));
    assert_eq!(
        json_str(&value, "node_id").as_deref(),
        Some(expected_node.as_str())
    );
    let expected = base64::engine::general_purpose::STANDARD.encode(payload);
    assert_eq!(
        json_str(&value, "payload_base64").as_deref(),
        Some(expected.as_str())
    );
    assert_eq!(
        json_str(value.get("payload").unwrap(), "event_id").as_deref(),
        Some("$dump")
    );
    assert_eq!(value.get("metadata").unwrap().encode(), "null");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn packs_dump_rejects_a_pack_selector_present_in_multiple_pools() {
    let dir = unique_temp_dir();
    let layout = DatabaseLayout::open(dir.clone()).unwrap();
    // Deterministic addresses sharing the leading hex digits `ab` in two
    // pools: a `0xab` prefix selector is ambiguous across them.
    for (shard_type, tail) in [(ShardType::EventDag, 1u8), (ShardType::State, 2u8)] {
        let pool = layout.pool_dir(shard_type).unwrap();
        let mut bytes = [0u8; mtxdb::packfile::PACK_ID_LEN];
        bytes[0] = 0xAB;
        bytes[mtxdb::packfile::PACK_ID_LEN - 1] = tail;
        let pack_id = PackId(bytes);
        let mut file = std::fs::File::create(pool.join(pack_id.filename())).unwrap();
        mtxdb::packfile::write_header_with_creation_seq(&mut file, &pack_id, 1).unwrap();
    }
    let cli = Cli {
        dirs: vec![dir.clone()],
        shard_type: None,
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Packs {
            action: super::PacksAction::List { all: true },
        },
    };
    let error = super::cmd_packs_dump(&cli, "0xab", None, None).unwrap_err();
    assert!(error.to_string().contains("ambiguous"), "{error}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn packs_dump_errors_when_the_pack_is_absent() {
    let dir = unique_temp_dir();
    DatabaseLayout::open(dir.clone()).unwrap();
    let cli = Cli {
        dirs: vec![dir.clone()],
        shard_type: Some(ShardType::EventDag),
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Packs {
            action: super::PacksAction::List { all: false },
        },
    };
    let error = super::cmd_packs_dump(&cli, "0x0", None, None).unwrap_err();
    assert!(error.to_string().contains("not found"), "{error}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn packs_extract_slices_target_collection_into_valid_pack() {
    let dir = unique_temp_dir();
    let layout = DatabaseLayout::open(dir.clone()).unwrap();
    let pool = layout.pool_dir(ShardType::EventDag).unwrap();
    let store = PackfileStorage::open(pool).unwrap();

    let room1_selector = "!room1:example.org";
    let room1_id = super::parse_collection_selector(room1_selector).unwrap();
    let room2_selector = "!room2:example.org";
    let room2_id = super::parse_collection_selector(room2_selector).unwrap();

    let node1_id = [0x11; 16];
    let node2_id = [0x22; 16];

    let payload1 = b"{\"event\":\"room1\"}";
    let digest1 = mtxdb::content_digest(DigestAlgorithm::Blake3, payload1);
    store
        .put_verified(
            &room1_id,
            &node1_id,
            &NodeData::new(Bytes::from_static(payload1)),
            &[0x01; 32],
            DigestAlgorithm::Blake3,
            &digest1,
            Some(b"timeline"),
        )
        .unwrap();

    let payload2 = b"{\"event\":\"room2\"}";
    let digest2 = mtxdb::content_digest(DigestAlgorithm::Blake3, payload2);
    store
        .put_verified(
            &room2_id,
            &node2_id,
            &NodeData::new(Bytes::from_static(payload2)),
            &[0x02; 32],
            DigestAlgorithm::Blake3,
            &digest2,
            Some(b"state"),
        )
        .unwrap();

    store.sync().unwrap();
    let pack_id = store.shard_summaries()[0].pack_id;
    drop(store);

    let cli = Cli {
        dirs: vec![dir.clone()],
        shard_type: Some(ShardType::EventDag),
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Packs {
            action: super::PacksAction::List { all: false },
        },
    };

    let out_pack = dir.join("room1_extracted.pack");
    super::cmd_packs_extract(&cli, &pack_id.to_string(), room1_selector, &out_pack).unwrap();

    assert!(out_pack.is_file());

    // Validate extracted pack
    let mut scanner = mtxdb::packfile::scan_records_iter(&out_pack).unwrap();
    let (offset, record) = scanner.next().unwrap().unwrap();
    assert_eq!(offset, mtxdb::packfile::HEADER_LEN as u64);
    assert_eq!(record.collection_id, room1_id);
    assert_eq!(record.hash, node1_id);
    assert_eq!(record.data.as_ref(), b"{\"event\":\"room1\"}");
    assert_eq!(
        record.metadata.as_ref().and_then(|m| m.role.as_deref()),
        Some(b"timeline".as_slice())
    );
    assert!(scanner.next().is_none());
    assert!(!scanner.torn_tail());

    // Validate header address stamped in dest pack: a fresh identity,
    // distinct from the source pack's.
    let mut file = std::fs::File::open(&out_pack).unwrap();
    let header = mtxdb::packfile::read_header(&mut file).unwrap().unwrap();
    assert_ne!(header.pack_id, pack_id);
    assert_eq!(
        header.pack_id.as_hex().len(),
        mtxdb::packfile::PACK_ID_LEN * 2
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// A temp database with one record synced into an `EventDag` pack, and a
/// `packs` CLI over it. Returns the directory, the CLI and the pack's id.
fn one_record_pack_fixture() -> (PathBuf, Cli, PackId) {
    let dir = unique_temp_dir();
    let layout = DatabaseLayout::open(dir.clone()).unwrap();
    let pool = layout.pool_dir(ShardType::EventDag).unwrap();
    let store = PackfileStorage::open(pool).unwrap();
    store
        .put(
            &[0x11; 16],
            &[0x22; 16],
            &NodeData::new(Bytes::from_static(b"data")),
        )
        .unwrap();
    store.sync().unwrap();
    let pack_id = store.shard_summaries()[0].pack_id;
    drop(store);

    let cli = Cli {
        dirs: vec![dir.clone()],
        shard_type: Some(ShardType::EventDag),
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Packs {
            action: super::PacksAction::List { all: false },
        },
    };
    (dir, cli, pack_id)
}

#[test]
fn packs_extract_rejects_same_source_and_dest() {
    let (dir, cli, pack_id) = one_record_pack_fixture();

    let source_path = DatabaseLayout::open(dir.clone())
        .unwrap()
        .pool_dir(ShardType::EventDag)
        .unwrap()
        .join(pack_id.filename());

    let error = super::cmd_packs_extract(
        &cli,
        &pack_id.to_string(),
        "0x11111111111111111111111111111111",
        &source_path,
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("cannot be the same file"),
        "{error}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn export_envelope_resolves_the_winning_frame_after_an_overwrite() {
    let dir = unique_temp_dir();
    let layout = DatabaseLayout::open(dir.clone()).unwrap();
    let pool_dir = layout.pool_dir(ShardType::EventDag).unwrap();

    let collection_id = [0x11; 16];
    let node_id = [0x22; 16];
    // Two versions of the same node, each with distinct payload, logical id,
    // and role. A small rotation threshold puts the second version in a
    // later (higher pack-id) shard, so it is the live winner the envelope
    // must report — and the one whose metadata must accompany the payload.
    let first = b"{ \"version\" : 1 }";
    let second = b"{ \"version\" : 2 }";

    {
        // One record per pack: the threshold admits a single frame after the
        // 4 KiB header, so each write rotates to a fresh shard.
        let store = PackfileStorage::open_with_max_shard_bytes(pool_dir.clone(), 4200).unwrap();
        let first_digest = mtxdb::content_digest(DigestAlgorithm::Blake3, first);
        store
            .put_verified(
                &collection_id,
                &node_id,
                &NodeData::new(Bytes::copy_from_slice(first)),
                &[0xA1; 32],
                DigestAlgorithm::Blake3,
                &first_digest,
                Some(b"v1"),
            )
            .unwrap();
        store.sync().unwrap();
        let second_digest = mtxdb::content_digest(DigestAlgorithm::Blake3, second);
        store
            .put_verified(
                &collection_id,
                &node_id,
                &NodeData::new(Bytes::copy_from_slice(second)),
                &[0xB2; 32],
                DigestAlgorithm::Blake3,
                &second_digest,
                Some(b"v2"),
            )
            .unwrap();
        store.sync().unwrap();
        assert!(
            store.shard_summaries().len() >= 2,
            "overwrite must land in a later pack to exercise winner resolution"
        );
    }

    // Drive the real export pipeline (scan -> winner selection -> frame
    // read) rather than the line-rendering helper in isolation.
    let store = PackfileStorage::open_read_only(pool_dir.clone()).unwrap();
    let pool = ShardPool::open_read_only(pool_dir).unwrap();
    let lines =
        super::export_lines(&store, &pool, &collection_id, super::ExportFormat::Envelope).unwrap();

    assert_eq!(lines.len(), 1, "one live record after physical dedup");
    let value = owned_value(std::str::from_utf8(&lines[0]).unwrap());
    let expected = base64::engine::general_purpose::STANDARD.encode(second);
    assert_eq!(
        json_str(&value, "payload_base64").as_deref(),
        Some(expected.as_str()),
        "winner must be the later frame's exact bytes"
    );
    assert_eq!(
        value
            .get("payload")
            .unwrap()
            .get("version")
            .unwrap()
            .encode(),
        "2"
    );
    let meta = value.get("metadata").expect("metadata object");
    let role = meta.get("role").expect("role object");
    assert_eq!(
        json_str(role, "value").as_deref(),
        Some("v2"),
        "metadata must come from the same winning frame as the payload"
    );
    assert!(
        json_str(meta, "logical_id").unwrap().starts_with("0xb2b2"),
        "logical id must be the winner's, not the overwritten version's"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn get_event_selector_matches_the_imported_record_id() {
    let template = default_matrix_import_template();
    let event = owned_value(
        r#"{"event_id":"$abc:example.org","room_id":"!r:example.org","type":"m.room.message"}"#,
    );
    let imported = template_node_id(&template, &event).unwrap().unwrap();
    assert_eq!(
        super::parse_get_id("$abc:example.org").unwrap(),
        imported,
        "`get $event_id` must resolve to the id import stored the record under"
    );
}

#[test]
fn get_dash_t_all_finds_record_across_pools() {
    let dir = unique_temp_dir();
    let layout = DatabaseLayout::open(dir.clone()).unwrap();
    let state_dir = layout.pool_dir(ShardType::State).unwrap();
    let store = PackfileStorage::open(state_dir).unwrap();
    let col_id = [0x01; 16];
    let node_id = [0x02; 16];
    let data = NodeData::new(Bytes::from_static(b"{\"hello\":\"world\"}\n"));
    store.put(&col_id, &node_id, &data).unwrap();
    store.sync().unwrap();

    let col_hex = format_id(&col_id);
    let node_hex = format_id(&node_id);

    let cli = Cli {
        dirs: vec![dir.clone()],
        shard_type: None,
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Get {
            collection: None,
            id: node_hex.clone(),
            raw: false,
            verbose: false,
            header: false,
            decode: None,
        },
    };

    cmd_get(&cli, None, &node_hex, false, false, false, None).unwrap();
    cmd_get(&cli, Some(&col_hex), &node_hex, true, false, false, None).unwrap();

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn get_and_scan_with_header_and_decode() {
    let dir = unique_temp_dir();
    let layout = DatabaseLayout::open(dir.clone()).unwrap();
    let state_dir = layout.pool_dir(ShardType::State).unwrap();
    let store = PackfileStorage::open(state_dir).unwrap();

    let canonical_id = b"sys:flat-kv";
    let col_id = derive_collection_id(Some(*b"INTL"), canonical_id);
    let genesis_meta = CollectionMetadata {
        member_namespace: Some(*b"INTL"),
        collection_canonical_id: canonical_id.to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Payload,
            digest_algorithm: DigestAlgorithm::Blake3,
        },
        payload: PayloadPolicy::Source,
        extension: None,
        role: Some("system_kv".to_owned()),
        schema: None,
    };

    store
        .ensure_collection_metadata(&col_id, &genesis_meta)
        .unwrap();

    let node_id = [0x42; 16];
    let data = NodeData::new(Bytes::from_static(b"{\"hello\":\"world\"}\n"));
    store.put(&col_id, &node_id, &data).unwrap();
    store.sync().unwrap();

    let col_hex = format_id(&col_id);
    let genesis_hex = format_id(&mtxdb::COLLECTION_METADATA_RECORD_ID);
    let node_hex = format_id(&node_id);

    let cli = Cli {
        dirs: vec![dir.clone()],
        shard_type: None,
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Get {
            collection: Some(col_hex.clone()),
            id: node_hex.clone(),
            raw: false,
            verbose: false,
            header: true,
            decode: Some("json".to_owned()),
        },
    };

    // Test get on data node with header and json decode
    cmd_get(
        &cli,
        Some(&col_hex),
        &node_hex,
        false,
        false,
        true,
        Some("json"),
    )
    .unwrap();

    // Test get on data node with raw decode
    cmd_get(
        &cli,
        Some(&col_hex),
        &node_hex,
        false,
        false,
        false,
        Some("raw"),
    )
    .unwrap();

    // Test get on genesis node with header
    cmd_get(&cli, Some(&col_hex), &genesis_hex, false, false, true, None).unwrap();

    // Test scan on collection with header and decode
    cmd_scan(
        &cli,
        &col_hex,
        false,
        true,
        Some("json"),
        -1,
        None,
        None,
        false,
        None,
        false,
        false,
    )
    .unwrap();

    // Test scan on collection with raw decode
    cmd_scan(
        &cli,
        &col_hex,
        false,
        false,
        Some("raw"),
        -1,
        None,
        None,
        false,
        None,
        false,
        false,
    )
    .unwrap();

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn info_dash_t_all_finds_collection_across_pools() {
    let dir = unique_temp_dir();
    let layout = DatabaseLayout::open(dir.clone()).unwrap();
    let state_dir = layout.pool_dir(ShardType::State).unwrap();
    let store = PackfileStorage::open(state_dir).unwrap();
    let col_id = [0x01; 16];
    let node_id = [0x02; 16];
    let data = NodeData::new(Bytes::from_static(b"hello"));
    store.put(&col_id, &node_id, &data).unwrap();
    store.sync().unwrap();

    let col_hex = format_id(&col_id);
    let cli = Cli {
        dirs: vec![dir.clone()],
        shard_type: None,
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Info {
            selector: Some(col_hex.clone()),
            pack: None,
            collection: None,
            stats: false,
        },
    };

    cmd_info(&cli, &col_hex, None, false).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn scan_dash_t_all_finds_collection_across_pools() {
    let dir = unique_temp_dir();
    let layout = DatabaseLayout::open(dir.clone()).unwrap();
    let state_dir = layout.pool_dir(ShardType::State).unwrap();
    let store = PackfileStorage::open(state_dir).unwrap();
    let col_id = [0x01; 16];
    let node_id = [0x02; 16];
    let data = NodeData::new(Bytes::from_static(b"hello"));
    store.put(&col_id, &node_id, &data).unwrap();
    store.sync().unwrap();

    let col_hex = format_id(&col_id);
    let cli = Cli {
        dirs: vec![dir.clone()],
        shard_type: None,
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Scan {
            selector: col_hex.clone(),
            verbose: false,
            header: false,
            decode: None,
            limit: -1,
            id: None,
            collection: None,
            raw: false,
            sort: None,
            reverse: false,
            live: false,
        },
    };

    cmd_scan(
        &cli, &col_hex, false, false, None, -1, None, None, false, None, false, false,
    )
    .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn live_collection_scan_excludes_superseded_versions() {
    let dir = unique_temp_dir();
    let layout = DatabaseLayout::open(dir.clone()).unwrap();
    let store = PackfileStorage::open(layout.pool_dir(ShardType::EventDag).unwrap()).unwrap();
    let collection = [0x41; 16];
    let node = [0x42; 16];
    store
        .put(
            &collection,
            &node,
            &NodeData::new(Bytes::from_static(b"first")),
        )
        .unwrap();
    store
        .put(
            &collection,
            &node,
            &NodeData::new(Bytes::from_static(b"second")),
        )
        .unwrap();
    store.sync().unwrap();
    assert_eq!(
        store
            .collection_live_locations(&collection)
            .unwrap()
            .unwrap()
            .len(),
        1,
        "the superseded version must not appear in the live index"
    );
    drop(store);

    let collection_hex = format_id(&collection);
    let cli = Cli {
        dirs: vec![dir.clone()],
        shard_type: Some(ShardType::EventDag),
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Scan {
            selector: collection_hex.clone(),
            verbose: false,
            header: false,
            decode: None,
            limit: 10,
            id: None,
            collection: None,
            raw: false,
            sort: None,
            reverse: false,
            live: true,
        },
    };
    let opts = ScanOptions {
        verbose: false,
        header: false,
        decode: None,
        limit: 10,
        node_id: None,
        raw: true,
        sort: None,
        reverse: false,
    };
    let (scanned_collection, rows, _) =
        cmd_scan_collection_live_rows(&cli, &collection_hex, &opts).unwrap();
    assert_eq!(scanned_collection, collection);
    let payloads: Vec<_> = rows
        .iter()
        .map(|(_, _, _, record)| record.as_ref().unwrap().data.as_ref())
        .collect();
    assert_eq!(payloads, [b"second".as_slice()]);
    cmd_scan(
        &cli,
        &collection_hex,
        false,
        false,
        None,
        10,
        None,
        None,
        false,
        None,
        false,
        true,
    )
    .unwrap();

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn state_collection_without_genesis_metadata_is_preserved_and_rejected() {
    let dir = unique_temp_dir();
    let layout = DatabaseLayout::open(dir.clone()).unwrap();
    let store = PackfileStorage::open(layout.pool_dir(ShardType::State).unwrap()).unwrap();

    let canonical_id = b"!room:example.org";
    let collection = derive_collection_id(Some(*b"STAT"), canonical_id);
    let node = [0x51; 16];
    store
        .put(
            &collection,
            &node,
            &NodeData::new(Bytes::from_static(b"old root")),
        )
        .unwrap();
    assert!(store
        .get_collection_metadata(&collection)
        .unwrap()
        .is_none());
    assert!(store.collection_exists(&collection));

    let metadata = CollectionMetadata {
        member_namespace: Some(*b"STAT"),
        collection_canonical_id: canonical_id.to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::ExternalCanonicalIdToCrosscheck,
            digest_algorithm: DigestAlgorithm::Sha256,
        },
        payload: PayloadPolicy::Source,
        extension: None,
        role: Some("state_hamt".to_owned()),
        schema: Some("sithnapse.state-hamt.v1".to_owned()),
    };
    let error = prepare_state_collection(&store, &collection, &metadata).unwrap_err();
    assert!(error.to_string().contains("refusing to reset"));
    assert!(store
        .get_collection_metadata(&collection)
        .unwrap()
        .is_none());
    assert_eq!(
        store.get(&collection, &node).unwrap().unwrap().bytes,
        Bytes::from_static(b"old root"),
        "unreferenced status cannot be established, so preserve the records"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

fn state_test_metadata(canonical_id: &[u8]) -> CollectionMetadata {
    CollectionMetadata {
        member_namespace: Some(*b"STAT"),
        collection_canonical_id: canonical_id.to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::ExternalCanonicalIdToCrosscheck,
            digest_algorithm: DigestAlgorithm::Sha256,
        },
        payload: PayloadPolicy::Source,
        extension: None,
        role: Some("state_hamt".to_owned()),
        schema: Some("sithnapse.state-hamt.v1".to_owned()),
    }
}

/// A server migration wrote HAMT roots with no genesis metadata, the server
/// stopped, and the import reopens the shared database. The reset must still
/// establish the collection, and the records written after it must be readable.
#[test]
fn state_collection_without_genesis_is_preserved_after_a_reopen() {
    let root = unique_temp_dir().join("state-reopen");
    let canonical_id = b"!room:example.org";
    let collection = derive_collection_id(Some(*b"STAT"), canonical_id);
    let old_node = [0x51; 16];

    {
        let db = Database::open(root.clone()).unwrap();
        let state = db.pool(ShardType::State);
        state
            .put(
                &collection,
                &old_node,
                &NodeData::new(Bytes::from_static(b"old root")),
            )
            .unwrap();
        state.sync().unwrap();
    }

    let db = Database::open(root.clone()).unwrap();
    let state = db.pool(ShardType::State);
    assert!(state
        .get_collection_metadata(&collection)
        .unwrap()
        .is_none());
    assert!(state.collection_exists(&collection));

    let metadata = state_test_metadata(canonical_id);
    let error = prepare_state_collection(state, &collection, &metadata).unwrap_err();
    assert!(error.to_string().contains("refusing to reset"));
    assert!(state
        .get_collection_metadata(&collection)
        .unwrap()
        .is_none());
    assert_eq!(
        state.get(&collection, &old_node).unwrap().unwrap().bytes,
        Bytes::from_static(b"old root")
    );

    drop(db);
    std::fs::remove_dir_all(root.parent().unwrap()).unwrap();
}

/// A room import prepares the collection, writes it, and a later import of the
/// same room prepares it again. The second prepare must leave the records alone.
#[test]
fn preparing_an_established_state_collection_again_keeps_its_records() {
    let root = unique_temp_dir().join("state-prepare-twice");
    let db = Database::open(root.clone()).unwrap();
    let state = db.pool(ShardType::State);
    let canonical_id = b"!room:example.org";
    let collection = derive_collection_id(Some(*b"STAT"), canonical_id);
    let node = [0x53; 16];
    let metadata = state_test_metadata(canonical_id);

    prepare_state_collection(state, &collection, &metadata).unwrap();
    state
        .put_many(
            &collection,
            &[(node, NodeData::new(Bytes::from_static(b"root")))],
        )
        .unwrap();
    prepare_state_collection(state, &collection, &metadata).unwrap();

    assert_eq!(
        state.get(&collection, &node).unwrap().unwrap().bytes,
        Bytes::from_static(b"root")
    );
    assert_eq!(
        state.get_collection_metadata(&collection).unwrap(),
        Some(metadata)
    );

    drop(db);
    std::fs::remove_dir_all(root.parent().unwrap()).unwrap();
}

/// Template whose record identity derives from `sender` rather than the
/// `event_id`, so two events with distinct `event_ids` can deliberately map to
/// the same storage node ID — modelling the truncated-hash collision the
/// import dedup must reject rather than collapse.
fn sender_identity_template() -> CollectionTemplate {
    CollectionTemplate {
        name: "collisions".into(),
        collection_kind: "federation".into(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Pointer {
                pointer: "/sender".into(),
            },
            digest_algorithm: DigestAlgorithm::Sha256,
        },
        payload: PayloadPolicy::Source,
        collection_key: CollectionKeyRule {
            pointer: "/room_id".into(),
            member_namespace: MATRIX_ROOM_MEMBER_NAMESPACE,
            display_id_pointer: "/room_id".into(),
        },
        establishment: None,
    }
}

#[allow(clippy::type_complexity)]
fn import_fixture(name: &str) -> (Database, PathBuf, PathBuf, CollectionTemplate, [u8; 16]) {
    let dir = unique_temp_dir().join(name);
    let db = Database::open(dir.clone()).unwrap();
    let template = sender_identity_template();
    let collection_id = template_collection_id(&template, "!room");
    let path = dir.join("fixture.json");
    (db, dir, path, template, collection_id)
}

fn import_event(event_id_: &str, sender: &str, room: &str) -> OwnedValue {
    owned_value(&format!(
        r#"{{"event_id":"{event_id_}","sender":"{sender}","room_id":"{room}","type":"m.room.message","content":{{"room_version":"11"}}}}"#
    ))
}

#[test]
fn imported_event_payload_uses_rezzy_redaction() {
    let event = owned_value(
        r#"{
            "event_id":"$event",
            "room_id":"!room",
            "sender":"@alice",
            "type":"m.room.message",
            "content":{"body":"hello","msgtype":"m.text"},
            "unsigned":{"age_ts":4},
            "__pdu_count":123,
            "__soft_failed":false
        }"#,
    );
    let bytes = redacted_event_bytes(&event, "11").unwrap();
    let json = bytes.to_vec();
    let redacted = OwnedValue::parse_bytes(&json).unwrap();
    assert!(redacted.get("unsigned").is_none());
    assert!(redacted.get("__pdu_count").is_none());
    assert!(redacted.get("__soft_failed").is_none());
    assert_eq!(
        redacted.get("content").and_then(|content| match content {
            OwnedValue::Object(fields) => Some(fields.is_empty()),
            _ => None,
        }),
        Some(true)
    );
}

#[test]
fn import_dedups_repeated_event_id_within_one_input() {
    let (db, _dir, path, template, collection_id) = import_fixture("import_dedup");
    let store = db.event_dag();
    let mut established = HashSet::new();
    established.insert(collection_id);
    let events = vec![
        import_event("$a", "@alice", "!room"),
        import_event("$a", "@alice", "!room"),
    ];
    import_pdu_events(
        &db,
        store,
        store,
        &path,
        &events,
        &[],
        None,
        None,
        &template,
        &mut established,
    )
    .unwrap();
    let stats = store.stats();
    assert_eq!(
        stats.put_many_calls, 2,
        "the event batch and the state-group batch must each use one write"
    );
    assert_eq!(
        stats.put_many_records, 2,
        "the repeated event_id must not be written twice, and its state mapping is one record"
    );
    let node_id = template_node_id(&template, &events[0]).unwrap().unwrap();
    assert!(
        store.get(&collection_id, &node_id).unwrap().is_some(),
        "the single stored record must be readable under its node ID"
    );
}

#[test]
fn import_rejects_two_event_ids_mapping_to_one_node_id() {
    let (db, _dir, path, template, collection_id) = import_fixture("import_input_collision");
    let store = db.event_dag();
    let mut established = HashSet::new();
    established.insert(collection_id);
    let events = vec![
        import_event("$a", "@alice", "!room"),
        import_event("$b", "@alice", "!room"),
    ];
    let error = import_pdu_events(
        &db,
        store,
        store,
        &path,
        &events,
        &[],
        None,
        None,
        &template,
        &mut established,
    )
    .expect_err("distinct event_ids truncating to one node ID must be rejected");
    assert!(
        error.to_string().contains("maps to multiple event IDs"),
        "saw: {error}"
    );
    assert_eq!(
        store.stats().put_many_calls,
        0,
        "no records may be written when the input itself collides"
    );
}

#[test]
fn import_rejects_cross_record_event_id_collision() {
    let (db, _dir, path, template, collection_id) = import_fixture("import_cross_record_collision");
    let store = db.event_dag();
    let existing = import_event("$c", "@alice", "!room");
    let node_id = template_node_id(&template, &existing).unwrap().unwrap();
    store
        .put(
            &collection_id,
            &node_id,
            &NodeData::new(Bytes::from(existing.encode().into_bytes())),
        )
        .unwrap();

    let incoming = import_event("$a", "@alice", "!room");
    let mut established = HashSet::new();
    established.insert(collection_id);
    let error = import_pdu_events(
        &db,
        store,
        store,
        &path,
        &[incoming],
        &[],
        None,
        None,
        &template,
        &mut established,
    )
    .expect_err("an on-disk record with a differing event_id must refuse to be overwritten");
    assert!(
        error
            .to_string()
            .contains("collides with a different event_id"),
        "saw: {error}"
    );
    assert_eq!(
        store.stats().put_many_calls,
        0,
        "a collision must not trigger a write"
    );
}

#[test]
fn import_keeps_existing_record_when_event_id_matches() {
    let (db, _, path, template, collection_id) = import_fixture("import_keep_same_event");
    let store = db.event_dag();
    // Keep the derived state-group index separate from the event store so
    // this assertion measures whether the existing event record was
    // rewritten.  Replaying the event may legitimately populate a
    // missing state-group mapping.
    let state_store = db.state().as_ref();
    let event = import_event("$a", "@alice", "!room");
    let node_id = template_node_id(&template, &event).unwrap().unwrap();
    store
        .put(
            &collection_id,
            &node_id,
            &NodeData::new(Bytes::from(event.clone().encode().into_bytes())),
        )
        .unwrap();

    let mut established = HashSet::new();
    established.insert(collection_id);
    import_pdu_events(
        &db,
        store,
        state_store,
        &path,
        &[event],
        &[],
        None,
        None,
        &template,
        &mut established,
    )
    .unwrap();
    assert_eq!(
        store.stats().put_many_calls,
        0,
        "a matching existing record must be classified present, not rewritten"
    );
}

#[test]
fn importer_loads_complete_cached_state_groups_without_recomputing() {
    let (db, _, path, template, collection_id) = import_fixture("import_cached_state_groups");
    let store = db.event_dag();
    let state_store = db.state().as_ref();
    let aux = mtxdb::auxiliary::AuxiliaryIndex::open(state_store, STATE_GROUP_NAMESPACE);
    aux.ensure_metadata().unwrap();
    let event = import_event("$cached", "@alice", "!room");
    let cached_group = [b'A'; STATE_GROUP_ID_LENGTH];
    aux.put(b"$cached", &cached_group).unwrap();
    let state_writes_before = state_store.stats().put_many_calls;
    let mut established = HashSet::new();
    established.insert(collection_id);
    import_pdu_events(
        &db,
        store,
        state_store,
        &path,
        std::slice::from_ref(&event),
        &[],
        None,
        None,
        &template,
        &mut established,
    )
    .unwrap();

    let loaded = load_state_groups(&aux, &["$cached".to_owned()]).unwrap();
    let StateGroupLoad::Complete(loaded) = loaded else {
        panic!("cached mapping should remain complete");
    };
    assert_eq!(loaded["$cached"], cached_group);
    assert_eq!(
        state_store.stats().put_many_calls,
        state_writes_before,
        "a complete cached mapping should not be recomputed or rewritten"
    );
}

#[test]
fn importer_ignores_the_previous_unversioned_state_group_cache() {
    const OLD_NAMESPACE: &str = "sys:matrix-state-groups";
    let (db, _, path, template, collection_id) =
        import_fixture("import_old_namespace_state_groups");
    let store = db.event_dag();
    let state_store = db.state().as_ref();
    let old_aux = mtxdb::auxiliary::AuxiliaryIndex::open(state_store, OLD_NAMESPACE);
    old_aux.ensure_metadata().unwrap();
    let stale = [b'A'; STATE_GROUP_ID_LENGTH];
    old_aux.put(b"$cached", &stale).unwrap();
    let new_aux = mtxdb::auxiliary::AuxiliaryIndex::open(state_store, STATE_GROUP_NAMESPACE);

    let event = import_event("$cached", "@alice", "!room");
    let mut established = HashSet::new();
    established.insert(collection_id);
    import_pdu_events(
        &db,
        store,
        state_store,
        &path,
        std::slice::from_ref(&event),
        &[],
        None,
        None,
        &template,
        &mut established,
    )
    .unwrap();

    // The versioned namespace cannot see the old entry, so the import recomputes and persists a
    // real instance id; the old namespace is left untouched as dead weight.
    let recomputed = new_aux
        .get(b"$cached")
        .unwrap()
        .expect("state-group mapping");
    assert_ne!(recomputed.as_slice(), stale.as_slice());
    assert!(valid_state_group_id(&recomputed));
    assert_eq!(old_aux.get(b"$cached").unwrap(), Some(stale.to_vec()));
}

#[test]
fn importer_repairs_unresolved_state_group_on_a_later_complete_import() {
    let (db, _, path, template, collection_id) = import_fixture("import_state_group_repair");
    let store = db.event_dag();
    let state_store = db.state().as_ref();
    let aux = mtxdb::auxiliary::AuxiliaryIndex::open(state_store, STATE_GROUP_NAMESPACE);
    aux.ensure_metadata().unwrap();
    let child = owned_value(
        r#"{"event_id":"$child","sender":"@alice","room_id":"!room","type":"m.room.message","prev_events":["$parent"],"content":{"room_version":"11"}}"#,
    );
    let mut established = HashSet::new();
    established.insert(collection_id);
    import_pdu_events(
        &db,
        store,
        state_store,
        &path,
        std::slice::from_ref(&child),
        &[],
        None,
        None,
        &template,
        &mut established,
    )
    .unwrap();
    assert_eq!(aux.get(b"$child").unwrap(), None);

    let repaired_parent = owned_value(
        r#"{"event_id":"$parent","sender":"@server","room_id":"!room","type":"m.room.create","content":{"room_version":"11"}}"#,
    );
    import_pdu_events(
        &db,
        store,
        state_store,
        &path,
        &[repaired_parent, child],
        &[],
        None,
        None,
        &template,
        &mut established,
    )
    .unwrap();
    let state_group = aux.get(b"$child").unwrap().expect("repaired mapping");
    assert!(valid_state_group_id(&state_group));
}

#[test]
fn template_node_id_matches_event_id_derivation() {
    let template = default_matrix_import_template();
    let event = raw_event(
        "!room_match",
        "$test_event:server.name",
        "m.room.message",
        "",
        "{}",
        "[]",
        "[]",
    );
    let from_event = template_node_id(&template, &event).unwrap().unwrap();
    let from_id = template_event_id_node_id(&template, "$test_event:server.name").unwrap();
    assert_eq!(
        from_event, from_id,
        "node ID derived from event ID string must match event node ID"
    );

    // Unsupported policy errors rather than silently returning None
    let mut custom = template.clone();
    custom.record_id_rule.policy = FrameIdPolicy::Payload;
    assert!(template_event_id_node_id(&custom, "$test_event:server.name").is_err());
    assert!(template_node_id(&custom, &event).is_err());
}

#[test]
fn search_reads_only_template_declared_fields() {
    let path = unique_temp_dir().join("search-template.json");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        &path,
        r#"{"search":{"fields":["/content/body"]},"record":{"identity":{"extract":{"path":"/event_id"}}}}"#,
    )
    .unwrap();
    let spec = search_template(Some(&path)).unwrap();
    let event = owned_value(
        r#"{"event_id":"$one","type":"m.room.message","content":{"body":"Hello, World"}}"#,
    );

    assert!(search_record_matches(&event, &spec.fields, "world"));
    assert!(!search_record_matches(
        &event,
        &spec.fields,
        "m.room.message"
    ));
    assert_eq!(spec.identity.as_deref(), Some("/event_id"));
    assert!(!valid_json_pointer("/content/~2body"));
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

/// The room's missing parents, derived the way a consumer would: every event
/// id its CSR rows reference whose body is stored nowhere. A body is stored in
/// the room's event collection, in the State pool's auth-chain collection, or
/// (for raw auth-chain events) as an auth projection.
fn missing_parents(
    db: &Database,
    template: &CollectionTemplate,
    room_id: &str,
) -> std::collections::BTreeSet<NodeId> {
    let nodes: Vec<NodeId> = RoomAuth::new(room_id)
        .referenced_event_ids(db)
        .unwrap()
        .iter()
        .map(|event_id| template_event_id_node_id(template, event_id).unwrap())
        .collect();
    let auth_collection =
        derive_collection_id(Some(mtxdb::MEMBER_NAMESPACE_AUTH), room_id.as_bytes());
    let places = [
        (db.event_dag(), template_collection_id(template, room_id)),
        (db.state(), auth_collection),
        (db.event_dag(), raw_auth_projection_collection_id(room_id)),
    ];
    let mut present = std::collections::HashSet::new();
    for (store, collection) in places {
        let found = store.get_many(&collection, &nodes).unwrap();
        present.extend(nodes.iter().zip(found).filter_map(|(n, f)| f.map(|_| *n)));
    }
    nodes.into_iter().filter(|n| !present.contains(n)).collect()
}

fn import_raw_events(
    db: &Database,
    template: &CollectionTemplate,
    room_id: &str,
    events: &[OwnedValue],
    auth_chain: &[OwnedValue],
) {
    let path = unique_temp_dir().join("room.jsonl");
    let mut established = HashSet::new();
    established.insert(template_collection_id(template, room_id));
    RawFixtureImporter {
        db,
        store: db.event_dag(),
        state_store: db.state().as_ref(),
        path: &path,
        template,
        established: &mut established,
    }
    .import(events, auth_chain);
}

#[test]
fn raw_import_marks_only_missing_parents_as_missing() {
    let db = Database::open(unique_temp_dir().join("raw_file_and_missing")).unwrap();
    let template = default_matrix_import_template();
    let create = raw_event(
        FIXTURE_ROOM,
        "$create",
        "m.room.create",
        "",
        r#"{"room_version":"11"}"#,
        "[]",
        "[]",
    );
    let child = raw_event(
        FIXTURE_ROOM,
        "$child",
        "m.room.message",
        "",
        "{}",
        r#"["$missing-prev"]"#,
        r#"["$same-file-auth"]"#,
    );
    let same_file_auth = raw_event(
        FIXTURE_ROOM,
        "$same-file-auth",
        "m.room.member",
        "@alice:test",
        r#"{"membership":"join"}"#,
        "[]",
        "[]",
    );

    import_raw_events(
        &db,
        &template,
        FIXTURE_ROOM,
        &[create, child, same_file_auth],
        &[],
    );

    let missing_node = template_event_id_node_id(&template, "$missing-prev").unwrap();
    assert_eq!(
        missing_parents(&db, &template, FIXTURE_ROOM),
        [missing_node].into(),
        "only the parent no file supplied is missing"
    );
    assert!(
        db.server_info().collection_ids().is_empty(),
        "missing parents are derived from the CSR, not stored"
    );
}

#[test]
fn raw_import_missing_parent_leaves_the_set_when_it_arrives() {
    let db = Database::open(unique_temp_dir().join("raw_waiting_arrival")).unwrap();
    let template = default_matrix_import_template();
    let create = raw_event(
        FIXTURE_ROOM,
        "$create",
        "m.room.create",
        "",
        r#"{"room_version":"11"}"#,
        "[]",
        "[]",
    );
    let child = raw_event(
        FIXTURE_ROOM,
        "$child",
        "m.room.message",
        "",
        "{}",
        r#"["$missing-prev"]"#,
        "[]",
    );
    let parent = raw_event(
        FIXTURE_ROOM,
        "$missing-prev",
        "m.room.message",
        "",
        "{}",
        "[]",
        "[]",
    );
    let parent_node = template_event_id_node_id(&template, "$missing-prev").unwrap();

    import_raw_events(&db, &template, FIXTURE_ROOM, &[create, child], &[]);
    assert!(missing_parents(&db, &template, FIXTURE_ROOM).contains(&parent_node));

    import_raw_events(&db, &template, FIXTURE_ROOM, &[parent], &[]);
    assert!(missing_parents(&db, &template, FIXTURE_ROOM).is_empty());
}

#[test]
fn raw_import_reimport_leaves_the_missing_set_unchanged() {
    let db = Database::open(unique_temp_dir().join("raw_idempotent")).unwrap();
    let template = default_matrix_import_template();
    let create = raw_event(
        FIXTURE_ROOM,
        "$create",
        "m.room.create",
        "",
        r#"{"room_version":"11"}"#,
        "[]",
        "[]",
    );
    let child = raw_event(
        FIXTURE_ROOM,
        "$child",
        "m.room.message",
        "",
        "{}",
        r#"["$missing-prev"]"#,
        "[]",
    );
    let parent_node = template_event_id_node_id(&template, "$missing-prev").unwrap();

    for _ in 0..3 {
        import_raw_events(
            &db,
            &template,
            FIXTURE_ROOM,
            &[create.clone(), child.clone()],
            &[],
        );
        assert_eq!(
            missing_parents(&db, &template, FIXTURE_ROOM),
            [parent_node].into()
        );
    }
}

#[test]
fn raw_import_derives_rejections_and_writes_flags() {
    use mtxdb::room_flag::*;

    let dir = unique_temp_dir().join("raw_auth");
    let db = Database::open(dir.clone()).unwrap();
    let template = default_matrix_import_template();
    let collection_id = template_collection_id(&template, "!room");
    let path = dir.join("room.jsonl");
    let store = db.event_dag();
    let state_store = db.state().as_ref();

    let event = |id: &str, kind: &str, state_key: &str, content: &str, prev: &str, auth: &str| {
        owned_value(&format!(
            r#"{{"event_id":"{id}","sender":"@alice:test","room_id":"!room","type":"{kind}","state_key":"{state_key}","content":{content},"prev_events":{prev},"auth_events":{auth}}}"#
        ))
    };
    let create = event(
        "$create",
        "m.room.create",
        "",
        r#"{"room_version":"11"}"#,
        "[]",
        "[]",
    );
    let join = event(
        "$join",
        "m.room.member",
        "@alice:test",
        r#"{"membership":"join"}"#,
        r#"["$create"]"#,
        r#"["$create"]"#,
    );
    let leave = event(
        "$leave",
        "m.room.member",
        "@alice:test",
        r#"{"membership":"leave"}"#,
        r#"["$join"]"#,
        r#"["$create","$join"]"#,
    );
    // Cites the stale join after Alice left, so check 5 rejects it.
    let stale = event(
        "$stale",
        "m.room.join_rules",
        "",
        r#"{"join_rule":"public"}"#,
        r#"["$leave"]"#,
        r#"["$create","$join"]"#,
    );
    let mut established = HashSet::new();
    established.insert(collection_id);
    import_pdu_events(
        &db,
        store,
        state_store,
        &path,
        &[create, join, leave],
        &[],
        None,
        None,
        &template,
        &mut established,
    )
    .unwrap();
    // The second file has to recover the earlier state from the store for
    // check 5 to see that Alice has left.
    import_pdu_events(
        &db,
        store,
        state_store,
        &path,
        &[stale],
        &[],
        None,
        None,
        &template,
        &mut established,
    )
    .unwrap();

    let flags = RoomFlagIndex::new(b"!room");
    assert_eq!(flags.count(&db, &FlagQuery::has(Flag::Present)).unwrap(), 4);
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Rejected)).unwrap(),
        1
    );

    let aux = mtxdb::auxiliary::AuxiliaryIndex::open(state_store, STATE_GROUP_NAMESPACE);
    let loaded = load_state_groups(&aux, &["$leave".to_owned(), "$stale".to_owned()]).unwrap();
    let StateGroupLoad::Complete(groups) = loaded else {
        panic!("state groups should be cached");
    };
    assert_eq!(
        groups["$stale"], groups["$leave"],
        "rejected state event must not change later state"
    );
}

fn raw_late_parent_fixture() -> [OwnedValue; 4] {
    [
        raw_event(
            "!room3",
            "$create",
            "m.room.create",
            "",
            r#"{"room_version":"11"}"#,
            "[]",
            "[]",
        ),
        raw_event(
            "!room3",
            "$join",
            "m.room.member",
            "@alice:test",
            r#"{"membership":"join"}"#,
            r#"["$create"]"#,
            r#"["$create"]"#,
        ),
        // Its parent is in the next file, so the first import cannot check it.
        raw_event(
            "!room3",
            "$stale",
            "m.room.join_rules",
            "",
            r#"{"join_rule":"public"}"#,
            r#"["$leave"]"#,
            r#"["$create","$join"]"#,
        ),
        raw_event(
            "!room3",
            "$leave",
            "m.room.member",
            "@alice:test",
            r#"{"membership":"leave"}"#,
            r#"["$join"]"#,
            r#"["$create","$join"]"#,
        ),
    ]
}

struct RawFixtureImporter<'a> {
    db: &'a Database,
    store: &'a PackfileStorage,
    state_store: &'a PackfileStorage,
    path: &'a Path,
    template: &'a CollectionTemplate,
    established: &'a mut HashSet<[u8; 16]>,
}

impl RawFixtureImporter<'_> {
    fn import(&mut self, events: &[OwnedValue], auth_chain: &[OwnedValue]) {
        import_pdu_events_with_raw(
            self.db,
            self.store,
            self.state_store,
            self.path,
            events,
            auth_chain,
            Some(&[]),
            None,
            None,
            self.template,
            self.established,
        )
        .unwrap();
    }
}

#[test]
fn raw_import_updates_earlier_verdicts_when_a_late_parent_arrives() {
    use mtxdb::room_flag::*;

    let dir = unique_temp_dir().join("raw_late_parent");
    let db = Database::open(dir.clone()).unwrap();
    let template = default_matrix_import_template();
    let collection_id = template_collection_id(&template, "!room3");
    let path = dir.join("room.jsonl");
    let store = db.event_dag();
    let state_store = db.state().as_ref();
    let [create, join, stale, leave] = raw_late_parent_fixture();

    let mut established = HashSet::new();
    established.insert(collection_id);
    let mut import = |events: &[OwnedValue]| {
        import_pdu_events(
            &db,
            store,
            state_store,
            &path,
            events,
            &[],
            None,
            None,
            &template,
            &mut established,
        )
        .unwrap();
    };
    import(&[create, join, stale]);

    let flags = RoomFlagIndex::new(b"!room3");
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Rejected)).unwrap(),
        0,
        "an event with a missing parent is left unchecked"
    );

    let record = mtxdb::interop::event_record_id(b"$stale");
    let verdict_collection = mtxdb::interop::event_verdict_collection_id(&record);
    let stale_verdict = || {
        stored(&db, mtxdb::ShardType::EventDag, verdict_collection, record)
            .map(|bytes| mtxdb::interop::decode_verdict(&bytes).unwrap().1.rejection)
    };
    assert_eq!(
        stale_verdict().flatten(),
        None,
        "the first file stores no rejection for the unchecked event"
    );

    import(&[leave]);

    assert_eq!(flags.count(&db, &FlagQuery::has(Flag::Present)).unwrap(), 4);
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Rejected)).unwrap(),
        1,
        "the late parent makes the earlier event fail check 5"
    );
    assert_eq!(
        flags.select(&db, &FlagQuery::has(Flag::Rejected)).unwrap(),
        vec![b"$stale".to_vec()],
        "the flag index must map the short id back to the event"
    );
    assert!(
        stale_verdict().is_some(),
        "the earlier event's stored verdict must be updated, not just its state group"
    );
}

#[test]
fn raw_import_tracks_missing_parents_across_files() {
    let db = Database::open(unique_temp_dir().join("raw_waiting_parents")).unwrap();
    let template = default_matrix_import_template();
    let [create, join, stale, leave] = raw_late_parent_fixture();
    let leave_node = template_event_id_node_id(&template, "$leave").unwrap();
    assert_eq!(
        template_event_id_node_id(&template, "$stale").unwrap(),
        template_node_id(&template, &stale).unwrap().unwrap(),
        "string and event node-id helpers must agree"
    );

    // $stale cites $leave as prev, and $leave is not supplied yet.
    let initial = [create, join, stale];
    import_raw_events(&db, &template, "!room3", &initial, &[]);
    assert_eq!(
        missing_parents(&db, &template, "!room3"),
        [leave_node].into()
    );

    // Re-importing the same file changes nothing.
    import_raw_events(&db, &template, "!room3", &initial, &[]);
    assert_eq!(
        missing_parents(&db, &template, "!room3"),
        [leave_node].into()
    );

    // The parent arriving clears it.
    import_raw_events(&db, &template, "!room3", &[leave], &[]);
    assert!(missing_parents(&db, &template, "!room3").is_empty());

    // A later child of a parent stored by an earlier file does not re-add it.
    let late = raw_event(
        "!room3",
        "$late",
        "m.room.message",
        "@alice:test",
        "{}",
        r#"["$leave"]"#,
        r#"["$create","$join"]"#,
    );
    import_raw_events(&db, &template, "!room3", &[late], &[]);
    assert!(missing_parents(&db, &template, "!room3").is_empty());
    assert!(db.server_info().collection_ids().is_empty());
}

const FEDERATION_FIRST_DOC: &str = r#"{
    "pdus": [
        {"event_id":"$create","sender":"@alice:test","room_id":"!room3",
         "type":"m.room.create","state_key":"",
         "content":{"creator":"@alice:test","room_version":"11"},
         "prev_events":[],"auth_events":[]},
        {"event_id":"$join","sender":"@alice:test","room_id":"!room3",
         "type":"m.room.member","state_key":"@alice:test",
         "content":{"membership":"join"},
         "prev_events":["$create"],"auth_events":["$create"]},
        {"event_id":"$stale","sender":"@alice:test","room_id":"!room3",
         "type":"m.room.join_rules","state_key":"",
         "content":{"join_rule":"public"},
         "prev_events":["$leave"],"auth_events":["$create","$join"]}
    ]
}"#;

const FEDERATION_SECOND_DOC: &str = r#"{
    "pdus": [
        {"event_id":"$late","sender":"@alice:test","room_id":"!room3",
         "type":"m.room.message","state_key":"",
         "content":{},"prev_events":["$leave"],
         "auth_events":["$create","$join"]}
    ],
    "auth_chain": [
        {"event_id":"$leave","sender":"@alice:test","room_id":"!room3",
         "type":"m.room.member","state_key":"@alice:test",
         "content":{"membership":"leave"},
         "prev_events":["$join"],"auth_events":["$create","$join"]}
    ]
}"#;

#[test]
fn raw_import_derives_missing_parents_for_federation_documents() {
    let dir = unique_temp_dir().join("raw_federation_dependencies");
    let db = Database::open(dir.clone()).unwrap();
    let template = default_matrix_import_template();
    let store = db.event_dag();
    let state_store = db.state().as_ref();
    let mut established = HashSet::new();

    let first = dir.join("first.json");
    std::fs::write(&first, FEDERATION_FIRST_DOC).unwrap();
    cmd_import_file(
        &db,
        store,
        state_store,
        &first,
        None,
        &template,
        &mut established,
        false,
        None,
        true,
    )
    .unwrap();

    let leave_node = template_event_id_node_id(&template, "$leave").unwrap();
    assert_eq!(
        missing_parents(&db, &template, "!room3"),
        [leave_node].into(),
        "federation PDUs must leave their missing parent derivable from the CSR"
    );

    let second = dir.join("second.json");
    std::fs::write(&second, FEDERATION_SECOND_DOC).unwrap();
    cmd_import_file(
        &db,
        store,
        state_store,
        &second,
        None,
        &template,
        &mut established,
        false,
        None,
        true,
    )
    .unwrap();

    assert!(
        missing_parents(&db, &template, "!room3").is_empty(),
        "an auth-chain parent in a federation document must not be missing"
    );
}

#[test]
fn raw_import_keeps_pre_v11_power_level_rejection_across_files() {
    use mtxdb::room_flag::*;

    let dir = unique_temp_dir().join("raw_v10_invite");
    let db = Database::open(dir.clone()).unwrap();
    let template = default_matrix_import_template();
    let collection_id = template_collection_id(&template, "!room5");
    let path = dir.join("room.jsonl");
    let store = db.event_dag();
    let state_store = db.state().as_ref();
    let [create, join_alice, power_levels, join_rules, join_bob, invite] = raw_v10_invite_fixture();

    let mut established = HashSet::new();
    established.insert(collection_id);
    let mut import = |events: &[OwnedValue]| {
        import_pdu_events(
            &db,
            store,
            state_store,
            &path,
            events,
            &[],
            None,
            None,
            &template,
            &mut established,
        )
        .unwrap();
    };
    import(&[
        create,
        join_alice,
        power_levels,
        join_rules,
        join_bob,
        invite,
    ]);

    let flags = RoomFlagIndex::new(b"!room5");
    let rejected = FlagQuery::has(Flag::Rejected);
    assert_eq!(flags.count(&db, &rejected).unwrap(), 1);
    assert_eq!(
        flags.select(&db, &rejected).unwrap(),
        vec![b"$invite".to_vec()]
    );

    let live = raw_event(
        "!room5",
        "$live",
        "m.room.message",
        "",
        "{}",
        r#"["$join_bob"]"#,
        r#"["$create","$join_alice"]"#,
    );
    import(&[live]);

    assert_eq!(
        flags.count(&db, &rejected).unwrap(),
        1,
        "a later file must not clear the pre-v11 invite rejection"
    );
    assert_eq!(
        flags.select(&db, &rejected).unwrap(),
        vec![b"$invite".to_vec()],
        "the rejected invite must still be selectable"
    );
}

#[test]
fn raw_import_does_not_project_hash_failing_content() {
    use mtxdb::room_flag::*;

    let dir = unique_temp_dir().join("raw_forged_projection");
    let db = Database::open(dir.clone()).unwrap();
    let template = default_matrix_import_template();
    let collection_id = template_collection_id(&template, "!room_forged");
    let path = dir.join("room.jsonl");
    let store = db.event_dag();
    let state_store = db.state().as_ref();
    let room = "!room_forged";

    let create = raw_event(
        room,
        "$create",
        "m.room.create",
        "",
        r#"{"room_version":"10"}"#,
        "[]",
        "[]",
    );
    let join_alice = raw_event(
        room,
        "$join_alice",
        "m.room.member",
        "@alice:test",
        r#"{"membership":"join"}"#,
        r#"["$create"]"#,
        r#"["$create"]"#,
    );
    let power_levels = forged_power_levels_with_stale_hash(room);
    let rules = raw_event(
        room,
        "$rules",
        "m.room.join_rules",
        "",
        r#"{"join_rule":"public"}"#,
        r#"["$pl"]"#,
        r#"["$create","$join_alice","$pl"]"#,
    );
    let join_bob = owned_value(
        r#"{"event_id":"$join_bob","sender":"@bob:test","room_id":"!room_forged","type":"m.room.member","state_key":"@bob:test","content":{"membership":"join"},"prev_events":["$rules"],"auth_events":["$create","$pl","$rules"]}"#,
    );

    let mut established = HashSet::new();
    established.insert(collection_id);
    let mut import = |events: &[OwnedValue]| {
        import_pdu_events(
            &db,
            store,
            state_store,
            &path,
            events,
            &[],
            None,
            None,
            &template,
            &mut established,
        )
        .unwrap();
    };
    import(&[create, join_alice, power_levels.clone(), rules, join_bob]);

    let node = template_node_id(&template, &power_levels).unwrap().unwrap();
    assert!(
        store
            .get(&raw_auth_projection_collection_id(room), &node)
            .unwrap()
            .is_none(),
        "a hash-failing event must not write an auth projection"
    );
    let loaded = load_collection_events(store, &collection_id, room).unwrap();
    let stored_pl = loaded
        .iter()
        .find(|event| event_id(event) == Some("$pl"))
        .expect("the power levels event is stored");
    assert!(
        stored_pl
            .get("content")
            .and_then(|content| content.get("invite"))
            .is_none(),
        "the forged invite field must not be overlaid from a projection"
    );

    let invite = owned_value(
        r#"{"event_id":"$invite","sender":"@bob:test","room_id":"!room_forged","type":"m.room.member","state_key":"@carol:test","content":{"membership":"invite"},"prev_events":["$join_bob"],"auth_events":["$create","$pl","$join_bob"]}"#,
    );
    import(&[invite]);

    let flags = RoomFlagIndex::new(room.as_bytes());
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Rejected)).unwrap(),
        0,
        "the redacted power levels invite level is 0, so Bob's invite must not be rejected"
    );
}

#[test]
fn raw_import_keeps_msc3389_power_level_rejection_across_files() {
    use mtxdb::room_flag::*;

    let dir = unique_temp_dir().join("raw_msc3389_invite");
    let db = Database::open(dir.clone()).unwrap();
    let template = default_matrix_import_template();
    let collection_id = template_collection_id(&template, "!room8");
    let path = dir.join("room.jsonl");
    let store = db.event_dag();
    let state_store = db.state().as_ref();

    let create = raw_event(
        "!room8",
        "$create",
        "m.room.create",
        "",
        r#"{"room_version":"org.matrix.msc3389.10"}"#,
        "[]",
        "[]",
    );
    let join_alice = raw_event(
        "!room8",
        "$join_alice",
        "m.room.member",
        "@alice:test",
        r#"{"membership":"join"}"#,
        r#"["$create"]"#,
        r#"["$create"]"#,
    );
    let power_levels = with_content_hash(
        raw_event(
            "!room8",
            "$pl",
            "m.room.power_levels",
            "",
            r#"{"users":{"@alice:test":100},"invite":50}"#,
            r#"["$join_alice"]"#,
            r#"["$create","$join_alice"]"#,
        ),
        "org.matrix.msc3389.10",
    );
    let join_rules = raw_event(
        "!room8",
        "$rules",
        "m.room.join_rules",
        "",
        r#"{"join_rule":"public"}"#,
        r#"["$pl"]"#,
        r#"["$create","$join_alice","$pl"]"#,
    );
    let join_bob = owned_value(
        r#"{"event_id":"$join_bob","sender":"@bob:test","room_id":"!room8","type":"m.room.member","state_key":"@bob:test","content":{"membership":"join"},"prev_events":["$rules"],"auth_events":["$create","$pl","$rules"]}"#,
    );
    let invite = owned_value(
        r#"{"event_id":"$invite","sender":"@bob:test","room_id":"!room8","type":"m.room.member","state_key":"@carol:test","content":{"membership":"invite"},"prev_events":["$join_bob"],"auth_events":["$create","$pl","$join_bob"]}"#,
    );

    let mut established = HashSet::new();
    established.insert(collection_id);
    let mut import = |events: &[OwnedValue]| {
        import_pdu_events(
            &db,
            store,
            state_store,
            &path,
            events,
            &[],
            None,
            None,
            &template,
            &mut established,
        )
        .unwrap();
    };
    import(&[
        create,
        join_alice,
        power_levels,
        join_rules,
        join_bob,
        invite,
    ]);

    let flags = RoomFlagIndex::new(b"!room8");
    let rejected = FlagQuery::has(Flag::Rejected);
    assert_eq!(
        flags.select(&db, &rejected).unwrap(),
        vec![b"$invite".to_vec()]
    );

    let live = raw_event(
        "!room8",
        "$live",
        "m.room.message",
        "",
        "{}",
        r#"["$join_bob"]"#,
        r#"["$create","$join_alice"]"#,
    );
    import(&[live]);

    assert_eq!(
        flags.select(&db, &rejected).unwrap(),
        vec![b"$invite".to_vec()],
        "MSC3389 uses v10 redaction and must keep the invite rejection"
    );
}

#[test]
fn raw_import_keeps_pre_v11_m_federate_rejection_across_files() {
    use mtxdb::room_flag::*;

    let dir = unique_temp_dir().join("raw_v10_federate");
    let db = Database::open(dir.clone()).unwrap();
    let template = default_matrix_import_template();
    let collection_id = template_collection_id(&template, "!room6");
    let path = dir.join("room.jsonl");
    let store = db.event_dag();
    let state_store = db.state().as_ref();

    let create = with_content_hash(
        raw_event(
            "!room6",
            "$create",
            "m.room.create",
            "",
            r#"{"room_version":"10","m.federate":false}"#,
            "[]",
            "[]",
        ),
        "10",
    );
    let join_alice = raw_event(
        "!room6",
        "$join_alice",
        "m.room.member",
        "@alice:test",
        r#"{"membership":"join"}"#,
        r#"["$create"]"#,
        r#"["$create"]"#,
    );
    let rules = raw_event(
        "!room6",
        "$rules",
        "m.room.join_rules",
        "",
        r#"{"join_rule":"public"}"#,
        r#"["$join_alice"]"#,
        r#"["$create","$join_alice"]"#,
    );
    let join_bob = owned_value(
        r#"{"event_id":"$join_bob","sender":"@bob:other","room_id":"!room6","type":"m.room.member","state_key":"@bob:other","content":{"membership":"join"},"prev_events":["$rules"],"auth_events":["$create","$rules"]}"#,
    );

    let mut established = HashSet::new();
    established.insert(collection_id);
    let mut import = |events: &[OwnedValue]| {
        import_pdu_events(
            &db,
            store,
            state_store,
            &path,
            events,
            &[],
            None,
            None,
            &template,
            &mut established,
        )
        .unwrap();
    };
    import(&[create, join_alice, rules]);
    import(&[join_bob]);

    let flags = RoomFlagIndex::new(b"!room6");
    let rejected = FlagQuery::has(Flag::Rejected);
    assert_eq!(
        flags.select(&db, &rejected).unwrap(),
        vec![b"$join_bob".to_vec()],
        "the redacted create event must keep m.federate=false for later files"
    );
}

#[test]
fn raw_auth_projection_keeps_v8_restricted_join_fields() {
    let member = owned_value(
        r#"{"event_id":"$join","sender":"@bob:test","room_id":"!room","type":"m.room.member","state_key":"@bob:test","content":{"membership":"join","join_authorised_via_users_server":"@alice:test"},"prev_events":[],"auth_events":[]}"#,
    );
    assert_eq!(
        raw_auth_projection(&member, "8")
            .unwrap()
            .get("join_authorised_via_users_server")
            .and_then(OwnedValue::as_str),
        Some("@alice:test")
    );
    assert!(raw_auth_projection(&member, "10")
        .and_then(|projection| projection.get("join_authorised_via_users_server").cloned())
        .is_none());

    let rules = owned_value(
        r#"{"event_id":"$rules","sender":"@alice:test","room_id":"!room","type":"m.room.join_rules","state_key":"","content":{"join_rule":"restricted","allow":[{"type":"m.room.member","room_id":"!other","state_key":"@alice:test"}]},"prev_events":[],"auth_events":[]}"#,
    );
    // `allow` is not read by the auth rules; the v8 redaction bug belongs
    // in rezzy, not in this projection.
    assert!(raw_auth_projection(&rules, "8").is_none());
    assert!(raw_auth_projection(&rules, "9").is_none());
    assert!(raw_auth_projection(&member, "11").is_none());

    // v11 redaction still strips all of m.room.third_party_invite.
    let third_party = owned_value(
        r#"{"event_id":"$tpi","sender":"@alice:test","room_id":"!room","type":"m.room.third_party_invite","state_key":"token","content":{"public_key":"key","public_keys":[{"public_key":"key"}]},"prev_events":[],"auth_events":[]}"#,
    );
    let projection = raw_auth_projection(&third_party, "11").unwrap();
    assert_eq!(
        projection.get("public_key").and_then(OwnedValue::as_str),
        Some("key")
    );
    assert!(projection.get("public_keys").is_some());

    let third_party_member = owned_value(
        r#"{"event_id":"$member","sender":"@bob:test","room_id":"!room","type":"m.room.member","state_key":"@bob:test","content":{"membership":"invite","third_party_invite":{"display_name":"Bob","signed":{"mxid":"@bob:test","token":"token"}}},"prev_events":[],"auth_events":[]}"#,
    );
    let projection = raw_auth_projection(&third_party_member, "10").unwrap();
    let token = projection.get("third_party_invite").unwrap();
    assert!(token.get("signed").is_some());
    assert!(token.get("display_name").is_none());

    let create = owned_value(
        r#"{"event_id":"$create","sender":"@alice:test","room_id":"!room","type":"m.room.create","state_key":"","content":{"room_version":"10","m.federate":false},"prev_events":[],"auth_events":[]}"#,
    );
    assert_eq!(
        raw_auth_projection(&create, "10")
            .unwrap()
            .get("m.federate")
            .and_then(OwnedValue::as_bool),
        Some(false)
    );

    // Version classification must come from rezzy's redaction rules, not a
    // numeric parse here. MSC3389 uses v10 redaction; MSC4242 and NutraCdo
    // use v11 redaction.
    let levels = owned_value(
        r#"{"event_id":"$pl","sender":"@alice:test","room_id":"!room","type":"m.room.power_levels","state_key":"","content":{"invite":50},"prev_events":[],"auth_events":[]}"#,
    );
    assert_eq!(
        raw_auth_projection(&levels, "org.matrix.msc3389.10")
            .unwrap()
            .get("invite")
            .and_then(OwnedValue::as_i64),
        Some(50)
    );
    assert_eq!(
        raw_auth_projection(&create, "org.matrix.msc3389.10")
            .unwrap()
            .get("m.federate")
            .and_then(OwnedValue::as_bool),
        Some(false)
    );
    assert!(raw_auth_projection(&create, "org.matrix.msc4242.12").is_none());
    assert!(raw_auth_projection(&create, "tk.nutra.cdo.12").is_none());

    // v11 redaction strips m.room.third_party_invite entirely; the projection
    // is what restores its public keys when a stored event is loaded back.
    let redacted = redacted_event_bytes(&third_party, "11").unwrap();
    let mut redacted = OwnedValue::parse_bytes(&redacted).unwrap();
    assert!(
        redacted
            .get("content")
            .and_then(|content| content.get("public_key"))
            .is_none(),
        "v11 redaction must strip the public key"
    );
    let projection = raw_auth_projection(&third_party, "11").unwrap();
    overlay_auth_projection(&mut redacted, projection);
    assert_eq!(
        redacted
            .get("content")
            .and_then(|content| content.get("public_key"))
            .and_then(OwnedValue::as_str),
        Some("key")
    );
}

#[test]
fn raw_auth_projection_rejects_conflicting_exports() {
    let dir = unique_temp_dir().join("projection_collision");
    let db = Database::open(dir).unwrap();
    let template = default_matrix_import_template();
    let store = db.event_dag();
    let room = "!projection_collision";
    let first = with_content_hash(
        raw_event(
            room,
            "$same",
            "m.room.power_levels",
            "",
            r#"{"invite":50}"#,
            "[]",
            "[]",
        ),
        "10",
    );
    let second = with_content_hash(
        raw_event(
            room,
            "$same",
            "m.room.power_levels",
            "",
            r#"{"invite":0}"#,
            "[]",
            "[]",
        ),
        "10",
    );
    write_raw_auth_projections(
        store,
        room,
        &template,
        "10",
        std::slice::from_ref(&first),
        &[],
    )
    .unwrap();
    let error =
        write_raw_auth_projections(store, room, &template, "10", &[second], &[]).unwrap_err();
    assert!(
        error.to_string().contains("$same")
            && error
                .to_string()
                .contains("conflicts with the stored projection"),
        "a verified projection conflict is corrupted data: {error}"
    );

    let node = template_node_id(&template, &first).unwrap().unwrap();
    let collection = raw_auth_projection_collection_id(room);
    let record = store.get(&collection, &node).unwrap().unwrap();
    let projection = OwnedValue::parse_bytes(&record.bytes).unwrap();
    assert_eq!(
        projection.get("invite").and_then(OwnedValue::as_i64),
        Some(50),
        "the first verified projection must survive a rejected conflict"
    );
}

#[test]
fn raw_import_keeps_v11_third_party_invite_public_keys_across_files() {
    let dir = unique_temp_dir().join("raw_v11_third_party_invite");
    let db = Database::open(dir.clone()).unwrap();
    let template = default_matrix_import_template();
    let collection_id = template_collection_id(&template, "!room7");
    let path = dir.join("room.jsonl");
    let store = db.event_dag();
    let state_store = db.state().as_ref();

    let create = raw_event(
        "!room7",
        "$create",
        "m.room.create",
        "",
        r#"{"room_version":"11"}"#,
        "[]",
        "[]",
    );
    let join_alice = raw_event(
        "!room7",
        "$join_alice",
        "m.room.member",
        "@alice:test",
        r#"{"membership":"join"}"#,
        r#"["$create"]"#,
        r#"["$create"]"#,
    );
    let third_party = with_content_hash(
        raw_event(
            "!room7",
            "$tpi",
            "m.room.third_party_invite",
            "token",
            r#"{"public_key":"key","public_keys":[{"public_key":"key"}]}"#,
            r#"["$join_alice"]"#,
            r#"["$create","$join_alice"]"#,
        ),
        "11",
    );
    let message = raw_event(
        "!room7",
        "$message",
        "m.room.message",
        "",
        r#"{"body":"after"}"#,
        r#"["$tpi"]"#,
        r#"["$create","$join_alice"]"#,
    );

    let mut established = HashSet::new();
    established.insert(collection_id);
    let mut import = |events: &[OwnedValue]| {
        import_pdu_events(
            &db,
            store,
            state_store,
            &path,
            events,
            &[],
            None,
            None,
            &template,
            &mut established,
        )
        .unwrap();
    };
    import(&[create, join_alice, third_party.clone()]);
    import(&[message]);

    let node = template_node_id(&template, &third_party).unwrap().unwrap();
    let stored = store.get(&collection_id, &node).unwrap().unwrap();
    let stored = OwnedValue::parse_bytes(&stored.bytes).unwrap();
    assert!(
        stored
            .get("content")
            .and_then(|content| content.get("public_key"))
            .is_none(),
        "v11 redaction still strips m.room.third_party_invite content"
    );

    let loaded = load_collection_events(store, &collection_id, "!room7").unwrap();
    let loaded_third_party = loaded
        .iter()
        .find(|event| event_id(event) == Some("$tpi"))
        .unwrap();
    assert_eq!(
        loaded_third_party
            .get("content")
            .and_then(|content| content.get("public_key"))
            .and_then(OwnedValue::as_str),
        Some("key"),
        "a later file must load the v11 third-party invite public key"
    );
    assert!(loaded_third_party
        .get("content")
        .and_then(|content| content.get("public_keys"))
        .is_some());
}

#[test]
fn raw_auth_projection_rejects_unknown_room_versions() {
    let dir = unique_temp_dir().join("projection_unknown_version");
    let db = Database::open(dir).unwrap();
    let template = default_matrix_import_template();
    let store = db.event_dag();
    let room = "!projection_unknown_version";
    let event = raw_event(
        room,
        "$create",
        "m.room.create",
        "",
        r#"{"room_version":"not-a-version","m.federate":false}"#,
        "[]",
        "[]",
    );

    let error = write_raw_auth_projections(store, room, &template, "not-a-version", &[event], &[])
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("not supported for auth projection"),
        "an unknown room version must be an error, got: {error}"
    );
}

#[test]
fn raw_import_rejects_unknown_room_version() {
    let dir = unique_temp_dir().join("raw_unknown_version");
    let db = Database::open(dir.clone()).unwrap();
    let template = default_matrix_import_template();
    let collection_id = template_collection_id(&template, "!room9");
    let path = dir.join("room.jsonl");
    let store = db.event_dag();
    let state_store = db.state().as_ref();
    let create = raw_event(
        "!room9",
        "$create",
        "m.room.create",
        "",
        r#"{"room_version":"not-a-version"}"#,
        "[]",
        "[]",
    );
    let mut established = HashSet::new();

    let error = import_pdu_events(
        &db,
        store,
        state_store,
        &path,
        &[create],
        &[],
        None,
        None,
        &template,
        &mut established,
    )
    .unwrap_err();
    let text = format!("{error:#}");
    assert!(
        text.contains("requires v4") || text.contains("not supported"),
        "an unknown room version must fail the import: {text}"
    );
    assert!(
        !established.contains(&collection_id),
        "a rejected room version must not establish the collection"
    );
}

#[test]
fn raw_import_keeps_soft_failed_flag_from_earlier_files() {
    use mtxdb::room_flag::*;

    let dir = unique_temp_dir().join("raw_soft_fail");
    let db = Database::open(dir.clone()).unwrap();
    let template = default_matrix_import_template();
    let collection_id = template_collection_id(&template, "!room4");
    let path = dir.join("room.jsonl");
    let store = db.event_dag();
    let state_store = db.state().as_ref();

    let create = raw_event(
        "!room4",
        "$create",
        "m.room.create",
        "",
        r#"{"room_version":"11"}"#,
        "[]",
        "[]",
    );
    let join = raw_event(
        "!room4",
        "$join",
        "m.room.member",
        "@alice:test",
        r#"{"membership":"join"}"#,
        r#"["$create"]"#,
        r#"["$create"]"#,
    );
    let mut soft = raw_event(
        "!room4",
        "$soft",
        "m.room.message",
        "",
        "{}",
        r#"["$join"]"#,
        r#"["$create","$join"]"#,
    );
    if let OwnedValue::Object(fields) = &mut soft {
        fields.insert("soft_failed".to_owned(), OwnedValue::Bool(true));
    }

    let mut established = HashSet::new();
    established.insert(collection_id);
    let mut import = |events: &[OwnedValue]| {
        import_pdu_events(
            &db,
            store,
            state_store,
            &path,
            events,
            &[],
            None,
            None,
            &template,
            &mut established,
        )
        .unwrap();
    };
    import(&[create, join, soft]);

    let flags = RoomFlagIndex::new(b"!room4");
    let soft_query = FlagQuery::has(Flag::SoftFailed);
    assert_eq!(flags.count(&db, &soft_query).unwrap(), 1);
    assert_eq!(
        flags.select(&db, &soft_query).unwrap(),
        vec![b"$soft".to_vec()]
    );

    let live = raw_event(
        "!room4",
        "$live",
        "m.room.message",
        "",
        "{}",
        r#"["$soft"]"#,
        r#"["$create","$join"]"#,
    );
    import(&[live]);

    assert_eq!(
        flags.count(&db, &soft_query).unwrap(),
        1,
        "a later file must not clear an earlier file's soft-fail flag"
    );
    assert_eq!(
        flags.select(&db, &soft_query).unwrap(),
        vec![b"$soft".to_vec()],
        "the soft-failed event must still be selectable"
    );
}

#[test]
fn raw_import_ignores_incoming_rejected_flags_on_valid_events() {
    let dir = unique_temp_dir().join("raw_unflagged");
    let db = Database::open(dir.clone()).unwrap();
    let template = default_matrix_import_template();
    let collection_id = template_collection_id(&template, "!room2");
    let path = dir.join("room.jsonl");
    let store = db.event_dag();
    let state_store = db.state().as_ref();

    let event = |id: &str, kind: &str, state_key: &str, content: &str, prev: &str, auth: &str| {
        owned_value(&format!(
            r#"{{"event_id":"{id}","sender":"@alice:test","room_id":"!room2","type":"{kind}","state_key":"{state_key}","content":{content},"prev_events":{prev},"auth_events":{auth}}}"#
        ))
    };
    let create = event(
        "$create",
        "m.room.create",
        "",
        r#"{"room_version":"11"}"#,
        "[]",
        "[]",
    );
    let join = event(
        "$join",
        "m.room.member",
        "@alice:test",
        r#"{"membership":"join"}"#,
        r#"["$create"]"#,
        r#"["$create"]"#,
    );
    // Valid join_rules event sent while Alice is joined, but carrying a false
    // incoming "rejected": true flag from an untrusted source.
    let mut valid_rules = event(
        "$rules",
        "m.room.join_rules",
        "",
        r#"{"join_rule":"public"}"#,
        r#"["$join"]"#,
        r#"["$create","$join"]"#,
    );
    if let OwnedValue::Object(fields) = &mut valid_rules {
        fields.insert("rejected".to_owned(), OwnedValue::Bool(true));
        fields.insert("__rejected".to_owned(), OwnedValue::Bool(true));
    }

    let mut established = HashSet::new();
    established.insert(collection_id);
    import_pdu_events(
        &db,
        store,
        state_store,
        &path,
        &[create, join, valid_rules],
        &[],
        None,
        None,
        &template,
        &mut established,
    )
    .unwrap();

    let aux = mtxdb::auxiliary::AuxiliaryIndex::open(state_store, STATE_GROUP_NAMESPACE);
    let loaded = load_state_groups(&aux, &["$join".to_owned(), "$rules".to_owned()]).unwrap();
    let StateGroupLoad::Complete(groups) = loaded else {
        panic!("state groups should be cached");
    };
    assert_ne!(
        groups["$rules"], groups["$join"],
        "valid event must not be excluded from state just because incoming JSON claimed rejected"
    );

    let flags = mtxdb::room_flag::RoomFlagIndex::new(b"!room2");
    assert_eq!(
        flags
            .count(
                &db,
                &mtxdb::room_flag::FlagQuery::has(mtxdb::room_flag::Flag::Rejected)
            )
            .unwrap(),
        0,
        "valid event must not be marked rejected"
    );
}

/// Open every pack in a pool and total the records it physically holds.
fn count_pack_records(pool_dir: &Path) -> u64 {
    let mut total = 0;
    if !pool_dir.exists() {
        return total;
    }
    for (pack_id, _, _) in glob_pack_files(pool_dir).unwrap() {
        let pack = pool_dir.join(pack_id.filename());
        let file = std::fs::File::open(pack).unwrap();
        let mut reader = std::io::BufReader::new(file);
        if mtxdb::packfile::read_header(&mut reader).unwrap().is_none() {
            continue;
        }
        while let Some(record) = mtxdb::packfile::read_record(&mut reader).unwrap() {
            let _ = record;
            total = total.saturating_add(1);
        }
    }
    total
}

#[test]
fn auth_chain_reimport_writes_nothing_new() {
    let root = unique_temp_dir();
    let pool_dir = root.join("pools").join("mtpl-event");
    let db = Database::open(root.clone()).unwrap();
    let store = db.event_dag();
    let input = root.join("export.json");
    std::fs::write(
        &input,
        r#"{
            "pdus": [
                {"event_id":"$c","room_id":"!room","sender":"@server",
                 "type":"m.room.create","state_key":"","content":{"creator":"@server","room_version":"10"},
                 "auth_events":[]}
            ],
            "auth_chain": [
                {"event_id":"$a1","room_id":"!room","sender":"@server",
                 "type":"m.room.member","state_key":"@server",
                 "content":{"membership":"join"},"auth_events":[]}
            ]
        }"#,
    )
    .unwrap();
    let template = default_matrix_import_template();
    let mut established = HashSet::new();

    cmd_import_file(
        &db,
        store,
        store,
        &input,
        None,
        &template,
        &mut established,
        false,
        None,
        true,
    )
    .unwrap();
    let auth_dir = pool_dir.parent().unwrap().join("mtpl-state");
    let first_import_records = count_pack_records(&auth_dir);
    let first_event_records = count_pack_records(&pool_dir);
    // Declares the create so batch_has_create resolves for any follow-up.
    cmd_import_file(
        &db,
        store,
        store,
        &input,
        None,
        &template,
        &mut established,
        false,
        None,
        true,
    )
    .unwrap();

    assert_eq!(
        count_pack_records(&auth_dir),
        first_import_records,
        "reimport must not append duplicate edge or auth-chain records"
    );
    assert_eq!(
        count_pack_records(&pool_dir),
        first_event_records,
        "reimport must not append duplicate verdicts or flag chunks"
    );
    let auth_store = db.state().as_ref();
    let auth_col = derive_collection_id(Some(mtxdb::MEMBER_NAMESPACE_AUTH), b"!room");
    let meta = auth_store
        .get_collection_metadata(&auth_col)
        .unwrap()
        .unwrap();
    assert_eq!(meta.role.as_deref(), Some("auth_chain"));
    assert_eq!(meta.collection_canonical_id, b"!room");
    assert!(meta.verify_collection_id(&auth_col));
}

#[test]
fn import_establishment_persists_the_matrix_extension() {
    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let store = db.event_dag();
    let input = root.join("export.json");
    // The create and a user record share one batch, so the import writes
    // metadata and records together. The genesis record must be written
    // first: with the old (records-then-metadata) order the engine now
    // rejects the late genesis write and the import fails.
    std::fs::write(
        &input,
        r#"{
            "pdus": [
                {"event_id":"$c","room_id":"!room","sender":"@server",
                 "type":"m.room.create","state_key":"",
                 "content":{"creator":"@server","room_version":"10"},
                 "auth_events":[]},
                {"event_id":"$m","room_id":"!room","sender":"@server",
                 "type":"m.room.message","content":{},
                 "auth_events":[]}
            ]
        }"#,
    )
    .unwrap();
    let template = default_matrix_import_template();
    let mut established = HashSet::new();
    cmd_import_file(
        &db,
        store,
        store,
        &input,
        None,
        &template,
        &mut established,
        false,
        None,
        true,
    )
    .unwrap();

    let collection_id = template_collection_id(&template, "!room");
    let metadata = store
        .get_collection_metadata(&collection_id)
        .unwrap()
        .expect("establishment must write the genesis metadata record");
    assert_eq!(metadata.collection_canonical_id, b"!room");
    assert_eq!(metadata.member_namespace, MATRIX_ROOM_MEMBER_NAMESPACE);
    let blob = metadata.extension.expect("Matrix room extension blob");

    // The self-describing blob is readable back through the same path
    // `info` uses, and carries the room version without touching a payload.
    let extension = matrix_room_extension_from_store(store, &collection_id)
        .expect("extension read back from the header");
    assert_eq!(extension.room_id.as_deref(), Some("!room"));
    assert_eq!(extension.create_event_id.as_deref(), Some("$c"));
    assert_eq!(extension.room_version.as_deref(), Some("10"));
    assert_eq!(extension.creator.as_deref(), Some("@server"));
    assert_eq!(extension.encode_blob(), blob, "blob is deterministic");
}

#[test]
fn pack_selectors_accept_only_pack_ids_and_deduplicate() {
    assert!(
        parse_pack_id_selector("3").is_err(),
        "decimal slots are not pack IDs"
    );
    assert!(
        parse_pack_id_selector("0x3-5").is_err(),
        "ranges are not pack IDs"
    );
    // Anything longer than the full address is rejected.
    assert!(parse_pack_id_selector(&format!(
        "0x{}",
        "a".repeat(mtxdb::packfile::PACK_ID_LEN * 2 + 2)
    ))
    .is_err());
    // A short prefix and a full 32-hex address are both accepted, and
    // exact duplicates are collapsed.
    let full = "ab".repeat(mtxdb::packfile::PACK_ID_LEN);
    let selectors = parse_pack_selectors(&[
        format!("0x{full}"),
        "0x0002".to_owned(),
        format!("0x{full}"),
    ])
    .unwrap();
    assert_eq!(selectors.len(), 2);
}

#[test]
fn event_room_id_reads_the_real_matrix_wire_key() {
    let event = owned_value(r#"{"room_id": "!abc:example.org", "type": "m.room.message"}"#);
    assert_eq!(event_room_id(&event), Some("!abc:example.org"));
}

#[test]
fn event_room_id_ignores_collection_id_and_missing_field() {
    // `collection_id` is mtxdb's own storage key, not a Matrix wire field —
    // event_room_id must not be fooled by it (regression for the room->collection rename).
    let event = owned_value(r#"{"collection_id": "deadbeef", "type": "m.room.message"}"#);
    assert_eq!(event_room_id(&event), None);

    assert_eq!(event_room_id(&owned_value("{}")), None);
    assert_eq!(event_room_id(&owned_value("42")), None);
    assert_eq!(
        event_room_id(&owned_value(r#"{"room_id": 42}"#)),
        None,
        "non-string room_id must not be coerced"
    );
}

#[test]
fn matrix_room_extension_reads_the_create_event() {
    let create = owned_value(
        r#"{
            "type": "m.room.create",
            "state_key": "",
            "event_id": "$create:example.org",
            "room_id": "!room:example.org",
            "content": {"creator": "@alice:example.org", "room_version": "10"}
        }"#,
    );
    let message =
        owned_value(r#"{"type":"m.room.message","event_id":"$m","room_id":"!room:example.org"}"#);
    let extension = MatrixRoomExtension::from_events(&[message, create]);
    assert_eq!(extension.room_id.as_deref(), Some("!room:example.org"));
    assert_eq!(
        extension.create_event_id.as_deref(),
        Some("$create:example.org")
    );
    assert_eq!(extension.room_version.as_deref(), Some("10"));
    assert_eq!(extension.creator.as_deref(), Some("@alice:example.org"));
}

#[test]
fn matrix_room_extension_round_trips_through_its_blob() {
    let extension = MatrixRoomExtension {
        room_id: Some("!room:example.org".into()),
        create_event_id: Some("$create".into()),
        room_version: Some("10".into()),
        creator: Some("@alice:example.org".into()),
        creation_ts: None,
    };
    let blob = extension.encode_blob();
    assert_eq!(MatrixRoomExtension::decode_blob(&blob), Some(extension));
    // A blob without the self-describing marker is rejected.
    assert_eq!(
        MatrixRoomExtension::decode_blob(br#"{"room_id":"!x"}"#),
        None
    );
}

/// Build a Synapse `event_json` mirror record: big-endian `i32`
/// `format_version`, big-endian `u32` `internal_metadata` length, then
/// the two JSON documents back to back with no delimiter.
fn encode_event_json_record(format_version: i32, internal_metadata: &str, json: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&format_version.to_be_bytes());
    buf.extend_from_slice(
        &u32::try_from(internal_metadata.len())
            .unwrap()
            .to_be_bytes(),
    );
    buf.extend_from_slice(internal_metadata.as_bytes());
    buf.extend_from_slice(json.as_bytes());
    buf
}

#[test]
fn decode_event_json_record_splits_metadata_from_pdu() {
    // The exact shape reported against a real event_json mirror record:
    // internal_metadata carrying a device_id, followed directly by the
    // PDU JSON with no delimiter between the two documents.
    let metadata = r#"{"device_id":"KIVSBMDOUC"}"#;
    let pdu = r#"{"signatures":{"test":{"ed25519:a_lPym":"sig"}},"unsigned":{"age_ts":4},"room_id":"!DZJveaTFXeqOdyrdeh:test","auth_events":[],"prev_events":[],"content":{"room_version":"11"},"depth":1,"hashes":{"sha256":"3sk9WqGW2B6W7tRNY5xwZ6qpHMiZX/hJWHBYx42uVyY"},"origin_server_ts":4,"sender":"@u1:test","state_key":"","type":"m.room.create"}"#;
    let record = encode_event_json_record(3, metadata, pdu);

    let (format_version, decoded_metadata, decoded_json) =
        decode_event_json_record(&record).expect("recognized event_json record shape");
    assert_eq!(format_version, 3);
    let decoded_metadata = String::from_utf8(decoded_metadata).unwrap();
    let decoded_json = String::from_utf8(decoded_json).unwrap();
    assert!(decoded_metadata.contains("KIVSBMDOUC"));
    assert!(decoded_json.contains("m.room.create"));

    // pretty_print_payload must reach the same decode, not the plain
    // single-JSON path (the whole record is not valid JSON on its own).
    let pretty = pretty_print_payload(&record).expect("event_json record is decodable");
    let pretty = String::from_utf8(pretty).unwrap();
    assert!(pretty.contains("format_version=3"));
    assert!(pretty.contains("KIVSBMDOUC"));
    assert!(pretty.contains("m.room.create"));
}

#[test]
fn decode_event_json_record_rejects_plain_json_and_garbage() {
    // A plain single JSON document is not this record shape — callers
    // must try `pretty_json_stream` first, not route ordinary payloads
    // through this decoder.
    assert!(decode_event_json_record(br#"{"type":"m.room.message"}"#).is_none());
    // Arbitrary short binary data, and a length field pointing past the
    // end of the buffer, must not panic and must not be mistaken for a
    // valid record.
    assert!(decode_event_json_record(b"\x00\x01").is_none());
    assert!(decode_event_json_record(&[0, 0, 0, 1, 0xFF, 0xFF, 0xFF, 0xFF, b'x']).is_none());
}

#[test]
fn matrix_room_extension_ignores_non_create_events() {
    // A stray "m.collection.create" (a leftover from a bad room->collection
    // rename) must never match — only the real Matrix wire event type does.
    let wrong_type = owned_value(
        r#"{"type":"m.collection.create","state_key":"","event_id":"$x","room_id":"!r"}"#,
    );
    let message = owned_value(r#"{"type":"m.room.message","event_id":"$m","room_id":"!r"}"#);
    let extension = MatrixRoomExtension::from_events(&[wrong_type, message]);
    assert_eq!(extension.create_event_id, None);
}

#[test]
fn matrix_batch_requires_a_valid_create_for_its_room() {
    let direct_create = owned_value(
        r#"{"type":"m.room.create","state_key":"","event_id":"$create","room_id":"!room:example.org"}"#,
    );
    let missing_state_key = owned_value(
        r#"{"type":"m.room.create","event_id":"$not-a-state-event","room_id":"!room:example.org"}"#,
    );
    let message = owned_value(
        r#"{"type":"m.room.message","event_id":"$message","room_id":"!room:example.org"}"#,
    );

    assert!(matrix_batch_has_create(
        &[direct_create, message.clone()],
        "!room:example.org"
    ));
    assert!(!matrix_batch_has_create(
        &[missing_state_key, message],
        "!room:example.org"
    ));
}

#[test]
fn matrix_batch_associates_a_room_id_less_create_through_auth() {
    let create = owned_value(r#"{"type":"m.room.create","state_key":"","event_id":"$create"}"#);
    let modern_auth = owned_value(
        r#"{"type":"m.room.message","event_id":"$message","room_id":"!room:example.org","auth_events":["$create"]}"#,
    );
    let legacy_auth = owned_value(
        r#"{"type":"m.room.message","event_id":"$legacy","room_id":"!room:example.org","auth_events":[["$create",{}]]}"#,
    );

    assert!(matrix_batch_has_create(
        &[create.clone(), modern_auth],
        "!room:example.org"
    ));
    assert!(matrix_batch_has_create(
        &[create, legacy_auth],
        "!room:example.org"
    ));
}

#[test]
fn import_admission_rejects_an_unestablished_room_before_writing() {
    let message = owned_value(
        r#"{"type":"m.room.message","event_id":"$message","room_id":"!room:example.org"}"#,
    );
    let template = default_matrix_import_template();
    let error = resolve_import_collection(&[message], None, None, &template, &HashSet::new())
        .expect_err("a room without a batch or persisted create must be rejected");
    assert!(error
        .to_string()
        .contains("no valid m.room.create event is in this input or already on disk"));
}

#[test]
fn import_admission_accepts_a_batch_that_establishes_its_room() {
    let create = owned_value(
        r#"{"type":"m.room.create","state_key":"","event_id":"$create","room_id":"!room:example.org","content":{"room_version":"10"}}"#,
    );
    let template = default_matrix_import_template();
    let resolved =
        resolve_import_collection(&[create], None, None, &template, &HashSet::new()).unwrap();
    assert_eq!(
        resolved.collection_id,
        template_collection_id(&template, "!room:example.org")
    );
    assert_eq!(resolved.canonical_id, "!room:example.org");
    assert!(resolved.batch_has_create);
}

#[test]
fn import_admission_rejects_a_pre_v4_room() {
    let template = default_matrix_import_template();
    // v3 is content-addressed but not URL-safe; v1/v2 are server-assigned.
    for version in ["1", "2", "3"] {
        let create = owned_value(&format!(
            r#"{{"type":"m.room.create","state_key":"","event_id":"$create","room_id":"!room:example.org","content":{{"room_version":"{version}"}}}}"#
        ));
        let error = resolve_import_collection(&[create], None, None, &template, &HashSet::new())
            .expect_err("a pre-v4 room must be rejected");
        assert!(
            error.to_string().contains("v4 or later"),
            "unexpected error for v{version}: {error}"
        );
    }
    // An absent room_version is v1 by the spec and must not be defaulted.
    let create = owned_value(
        r#"{"type":"m.room.create","state_key":"","event_id":"$create","room_id":"!room:example.org"}"#,
    );
    let error = resolve_import_collection(&[create], None, None, &template, &HashSet::new())
        .expect_err("an absent room_version must be rejected, not defaulted to v1");
    assert!(
        error.to_string().contains("v4 or later"),
        "unexpected error: {error}"
    );
}

#[test]
fn import_admission_resolves_v12_collections_from_the_create_event_id() {
    let template = default_matrix_import_template();
    // Room version 12 derives room identity from the accepted create
    // event, so a create without `room_id` still establishes its
    // collection. The canonical id is normalized to the `!<hash>` room id
    // ordinary v12 events carry (MSC4291), not the raw `$<hash>` id.
    let create = owned_value(
        r#"{"type":"m.room.create","state_key":"","event_id":"$v12create","content":{"room_version":"12"}}"#,
    );
    let resolved =
        resolve_import_collection(&[create], None, None, &template, &HashSet::new()).unwrap();
    assert_eq!(resolved.canonical_id, "!v12create");
    assert_eq!(
        resolved.collection_id,
        template_collection_id(&template, "!v12create")
    );
    assert!(resolved.batch_has_create);
}

#[test]
fn import_admission_reuses_v12_create_identity_for_followup_events() {
    let template = default_matrix_import_template();
    let canonical_id = "!v12create";
    let collection_id = template_collection_id(&template, canonical_id);
    let established = HashSet::from([collection_id]);
    // Ordinary v12 events reference the room as `!<create-id>` (MSC4291).
    let message =
        owned_value(r#"{"type":"m.room.message","event_id":"$message","room_id":"!v12create"}"#);

    let resolved =
        resolve_import_collection(&[message], None, None, &template, &established).unwrap();

    assert_eq!(resolved.canonical_id, canonical_id);
    assert_eq!(resolved.collection_id, collection_id);
    assert!(!resolved.batch_has_create);
}

/// A v12 event id in the real wire shape: SHA-256 of the event, URL-safe
/// unpadded base64 (v12's `ReferenceHashEncoding`), prefixed with `$`.
///
/// This stands in for the full reference hash, which is taken over the
/// redacted canonical event; it produces an id with the correct alphabet
/// so the sigil-normalization path is exercised with realistic data.
fn v12_reference_event_id(event: &OwnedValue) -> String {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    let digest = content_digest(DigestAlgorithm::Sha256, &event.encode().into_bytes());
    format!("${}", URL_SAFE_NO_PAD.encode(digest))
}

#[test]
fn import_v12_followup_batch_lands_in_the_create_collection() {
    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let store = db.event_dag();
    let template = default_matrix_import_template();
    let mut established = HashSet::new();

    // A v12 event id is a SHA-256 reference hash and its room id is that
    // id with the `$` sigil swapped for `!` (MSC4291). Hash the create
    // event and derive both forms through the real policy rather than a
    // hand-written placeholder id.
    let create_event = owned_value(
        r#"{"event_id":"","sender":"@server","type":"m.room.create","state_key":"","content":{"creator":"@server","room_version":"12"},"auth_events":[]}"#,
    );
    let create_id = v12_reference_event_id(&create_event);
    let room_id = MatrixRoomVersion::V12.normalize_collection_identity(&create_id);

    // Batch 1: the v12 create event alone. It carries no `room_id`; the
    // collection identity is the create event's id, normalized to the
    // `!<hash>` room-id form ordinary events reference.
    let create_path = root.join("create.json");
    std::fs::write(
        &create_path,
        format!(
            r#"{{"pdus":[{{"event_id":"{create_id}","sender":"@server","type":"m.room.create","state_key":"","content":{{"creator":"@server","room_version":"12"}},"auth_events":[]}}]}}"#
        ),
    )
    .unwrap();
    cmd_import_file(
        &db,
        store,
        store,
        &create_path,
        None,
        &template,
        &mut established,
        false,
        None,
        true,
    )
    .unwrap();

    let collection_id = template_collection_id(&template, &room_id);
    assert!(
        store
            .get_collection_metadata(&collection_id)
            .unwrap()
            .is_some(),
        "a v12 create must establish a collection keyed by the normalized room id"
    );

    // Batch 2: a separate ordinary v12 event. It references the room as
    // `!<create-id>` and must resolve to the same collection rather than
    // spawn a second one.
    let message_value = owned_value(&format!(
        r#"{{"event_id":"$m1","room_id":"{room_id}","sender":"@server","type":"m.room.message","content":{{}},"auth_events":[]}}"#
    ));
    let message_path = root.join("message.json");
    std::fs::write(
        &message_path,
        format!(
            r#"{{"pdus":[{{"event_id":"$m1","room_id":"{room_id}","sender":"@server","type":"m.room.message","content":{{}},"auth_events":[]}}]}}"#
        ),
    )
    .unwrap();
    cmd_import_file(
        &db,
        store,
        store,
        &message_path,
        None,
        &template,
        &mut established,
        false,
        None,
        true,
    )
    .unwrap();

    let node_id = template_node_id(&template, &message_value)
        .unwrap()
        .unwrap();
    assert!(
        store.get(&collection_id, &node_id).unwrap().is_some(),
        "the follow-up event must be stored under the create event's collection"
    );
}

#[test]
fn import_real_v12_room_slice_uses_the_normalized_collection_identity() {
    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let store = db.event_dag();
    let template = default_matrix_import_template();
    let mut established = HashSet::new();

    // A real v12 room exported from the external dag-toolkit corpus: a
    // DAG-complete prefix (depth 1..20) rooted at the create event, so
    // every `prev_events`/`auth_events` reference resolves inside the
    // batch. The create id is `$kgoc...` and ordinary events reference
    // `!kgoc...` -- MSC4291's room-id form -- which the importer must
    // normalize back to the create's identity.
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/v12-room-slice.jsonl");
    cmd_import_file(
        &db,
        store,
        store,
        &fixture,
        None,
        &template,
        &mut established,
        false,
        None,
        true,
    )
    .unwrap();

    let create_id = "$kgoc2ebwy1GOVtzOr5_-tXVTvzSHgtGaMSI_i59vogs";
    let room_id = "!kgoc2ebwy1GOVtzOr5_-tXVTvzSHgtGaMSI_i59vogs";
    assert_eq!(
        MatrixRoomVersion::V12.normalize_collection_identity(create_id),
        room_id,
        "the fixture's room id must be the create id with the sigil swapped"
    );

    let collection_id = template_collection_id(&template, room_id);
    assert!(
        store
            .get_collection_metadata(&collection_id)
            .unwrap()
            .is_some(),
        "the real v12 room must land in the create event's normalized collection"
    );

    // Every fixture PDU is retained, plus the collection's genesis
    // metadata record. State groups live in a separate auxiliary index,
    // so they do not inflate this room collection's entry count.
    let fixture_events = std::fs::read_to_string(&fixture)
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count();
    let (entries, _, _) = store
        .collection_index_info(&collection_id)
        .expect("collection index must be present after import");
    assert_eq!(
        entries,
        fixture_events + 1,
        "the collection must retain all {fixture_events} fixture events plus its metadata record"
    );

    assert_eq!(
        established,
        HashSet::from([collection_id]),
        "the whole slice must resolve to exactly one collection"
    );
}

#[test]
fn collection_canonical_id_honours_the_template_membership_pointer() {
    let mut template = sender_identity_template();
    template.collection_key.pointer = "/scope".into();
    let events = vec![owned_value(r#"{"event_id":"$a","scope":"tenant-a"}"#)];
    assert_eq!(
        collection_canonical_id(&events, &template)
            .unwrap()
            .as_deref(),
        Some("tenant-a")
    );
    // Events disagreeing on the membership value are rejected, not
    // silently coalesced into one collection.
    let mixed = vec![
        owned_value(r#"{"event_id":"$a","scope":"tenant-a"}"#),
        owned_value(r#"{"event_id":"$b","scope":"tenant-b"}"#),
    ];
    assert!(collection_canonical_id(&mixed, &template).is_err());
}

#[test]
fn checked_in_matrix_template_matches_the_executable_importer() {
    let template_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../templates/matrix-event-v1.json");
    let compiled = compile_import_template(Some(&template_path))
        .expect("checked-in Matrix template must be executable");
    assert_eq!(compiled, default_matrix_import_template());
}

#[test]
fn compile_import_template_rejects_an_unsupported_digest_algorithm() {
    let template = br#"{
        "format": "mtxdb.collection-template/v1",
        "name": "matrix-event-v1",
        "record": {
            "identity": {
                "extract": {"kind": "json-pointer-rfc-6901", "path": "/event_id"},
                "internal_key": {"algorithm": "sha256-truncated"}
            }
        },
        "collection": {
            "membership": {"extract": {"kind": "json-pointer-rfc-6901", "path": "/room_id"}}
        },
        "establishment": {"selector": "type == m.room.create && state_key == ''"}
    }"#;
    let error = compile_reject(template).expect_err(
        "an unsupported digest algorithm must be rejected at compile time, not mid-import",
    );
    assert!(
        format!("{error:#}").contains("unsupported internal-key algorithm"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn compile_import_template_rejects_a_projection_payload_policy() {
    let template = br#"{
        "format": "mtxdb.collection-template/v1",
        "name": "matrix-event-v1",
        "record": {
            "identity": {
                "extract": {"kind": "json-pointer-rfc-6901", "path": "/event_id"},
                "internal_key": {"algorithm": "sha2-256"}
            },
            "payload": {"policy": "projection", "include": ["/type", "/sender"]}
        },
        "collection": {
            "membership": {"extract": {"kind": "json-pointer-rfc-6901", "path": "/room_id"}}
        },
        "establishment": {"selector": "type == m.room.create && state_key == ''"}
    }"#;
    let error = compile_reject(template).expect_err(
        "a projection payload policy must be rejected instead of degrading to full-source retention",
    );
    assert!(
        format!("{error:#}").contains(r#"payload policy "projection" is not supported"#),
        "unexpected error: {error:#}"
    );
}

#[test]
fn compile_import_template_rejects_a_non_string_payload_policy() {
    let template = br#"{
        "format": "mtxdb.collection-template/v1",
        "name": "matrix-event-v1",
        "record": {
            "identity": {
                "extract": {"kind": "json-pointer-rfc-6901", "path": "/event_id"},
                "internal_key": {"algorithm": "sha2-256"}
            },
            "payload": {"policy": 42}
        },
        "collection": {
            "membership": {"extract": {"kind": "json-pointer-rfc-6901", "path": "/room_id"}}
        },
        "establishment": {"selector": "type == m.room.create && state_key == ''"}
    }"#;
    let error = compile_reject(template).expect_err(
        "a present non-string payload policy must be rejected instead of silently compiling to source retention",
    );
    assert!(
        format!("{error:#}").contains("payload policy must be a string"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn compile_import_template_rejects_a_distinct_display_label() {
    let template = br#"{
        "format": "mtxdb.collection-template/v1",
        "name": "matrix-event-v1",
        "record": {
            "identity": {
                "extract": {"kind": "json-pointer-rfc-6901", "path": "/event_id"},
                "internal_key": {"algorithm": "sha2-256"}
            },
            "payload": {"policy": "retain-source"}
        },
        "collection": {
            "membership": {"extract": {"kind": "json-pointer-rfc-6901", "path": "/room_id"}},
            "labels": [{"name": "display_id", "value": "/canonical_alias"}]
        },
        "establishment": {"selector": "type == m.room.create && state_key == ''"}
    }"#;
    let error = compile_reject(template).expect_err(
        "a distinct display label must be rejected since the importer cannot persist it",
    );
    assert!(
        format!("{error:#}").contains("display label"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn compile_import_template_rejects_non_array_labels() {
    let template = br#"{
        "format": "mtxdb.collection-template/v1",
        "name": "matrix-event-v1",
        "record": {
            "identity": {
                "extract": {"kind": "json-pointer-rfc-6901", "path": "/event_id"},
                "internal_key": {"algorithm": "sha2-256"}
            },
            "payload": {"policy": "retain-source"}
        },
        "collection": {
            "membership": {"extract": {"kind": "json-pointer-rfc-6901", "path": "/room_id"}},
            "labels": {"name": "display_id", "value": "membership-value"}
        },
        "establishment": {"selector": "type == m.room.create && state_key == ''"}
    }"#;
    let error = compile_reject(template)
        .expect_err("a present non-array collection.labels must not be silently treated as absent");
    assert!(
        format!("{error:#}").contains("collection.labels must be an array"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn compile_import_template_rejects_a_malformed_display_label() {
    for label_value in [
        r#"{"kind":"json-pointer-rfc-6901"}"#,
        r#"{"path": 42}"#,
        r"42",
        r"null",
        r#"["/canonical_alias"]"#,
        r"true",
    ] {
        let template = format!(
            r#"{{
            "format": "mtxdb.collection-template/v1",
            "name": "matrix-event-v1",
            "record": {{
                "identity": {{
                    "extract": {{"kind": "json-pointer-rfc-6901", "path": "/event_id"}},
                    "internal_key": {{"algorithm": "sha2-256"}}
                }},
                "payload": {{"policy": "retain-source"}}
            }},
            "collection": {{
                "membership": {{"extract": {{"kind": "json-pointer-rfc-6901", "path": "/room_id"}}}},
                "labels": [{{"name": "display_id", "value": {label_value}}}]
            }},
            "establishment": {{"selector": "type == m.room.create && state_key == ''"}}
        }}"#
        );
        let error = compile_reject(template.as_bytes()).expect_err(
            "a malformed display label value must not silently fall back to the membership pointer",
        );
        assert!(
            format!("{error:#}").contains("display label"),
            "unexpected error for label value {label_value}: {error:#}"
        );
    }
}

#[test]
fn pointer_extraction_supports_array_indices_and_rejects_bad_pointers() {
    let event =
        owned_value(r#"{"auth_events": ["$a", "$b"], "event_id": "$c", "a~1b": "escaped"}"#);
    // Valid array index.
    assert_eq!(extract_pointer_string(&event, "/auth_events/1"), Some("$b"));
    // Valid tilde escaping: ~0 → ~, ~1 → /.
    assert_eq!(extract_pointer_string(&event, "/a~01b"), Some("escaped"));
    // A non-empty pointer must start with `/`; a bare field name is not
    // a valid RFC 6901 pointer and must not silently resolve anything.
    assert_eq!(extract_pointer_string(&event, "event_id"), None);
    // Out-of-range and non-numeric array segments must not panic or
    // fall through to some other value.
    assert_eq!(extract_pointer_string(&event, "/auth_events/9"), None);
    assert_eq!(extract_pointer_string(&event, "/auth_events/x"), None);
    // Leading zeroes in array indices are forbidden by RFC 6901.
    assert_eq!(extract_pointer_string(&event, "/auth_events/01"), None);
    assert_eq!(extract_pointer_string(&event, "/auth_events/00"), None);
    // Signs and non-digit characters are forbidden by RFC 6901.
    assert_eq!(extract_pointer_string(&event, "/auth_events/+1"), None);
    assert_eq!(extract_pointer_string(&event, "/auth_events/-1"), None);
    // Invalid tilde escapes (~2, ~a, trailing ~) are forbidden.
    assert_eq!(extract_pointer_string(&event, "/a~2b"), None);
    assert_eq!(extract_pointer_string(&event, "/a~ab"), None);
    assert_eq!(extract_pointer_string(&event, "/a~"), None);
}

// `physical_layout`/`avoidable_spread_bytes` coverage now lives with
// their implementation in `mtxdb::packfile::layout::tests` —
// this crate just imports and displays them, nothing left to test here.

// ---- HAMT decoder tests ----

/// Encode a length-prefixed UTF-8 string (u32 LE length + bytes).
fn encode_lps(s: &str) -> Vec<u8> {
    let len = u32::try_from(s.len()).unwrap();
    let cap = usize::try_from(len).unwrap().checked_add(s.len()).unwrap();
    let mut out = Vec::with_capacity(cap);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(s.as_bytes());
    out
}

/// Matches `synapse/rust/src/state_hamt.rs`'s
/// `serde_json::to_string(&(event_type, state_key))` -- the real K
/// encoding is a single JSON 2-element array string, not a raw tuple
/// codec.
fn hamt_leaf_key_json(event_type: &str, state_key: &str) -> String {
    OwnedValue::Array(vec![
        OwnedValue::from(event_type),
        OwnedValue::from(state_key),
    ])
    .encode()
}

/// Build a rezzy wire-v1 HAMT node from leaf triples and child hashes.
///
/// Each leaf is encoded as two length-prefixed strings matching the real
/// Synapse format: K = `serde_json::to_string((event_type, state_key))`,
/// V = `event_id`.
fn build_hamt_node(leaves: &[(&str, &str, &str)], child_hashes: &[[u8; 32]]) -> Vec<u8> {
    let mut datamap: u32 = 0;
    let mut nodemap: u32 = 0;
    for i in 0..leaves.len() {
        datamap |= 1u32.checked_shl(u32::try_from(i).unwrap()).unwrap_or(0);
    }
    let leaf_count = u32::try_from(leaves.len()).unwrap();
    let child_count = u32::try_from(child_hashes.len()).unwrap();
    for i in 0..child_hashes.len() {
        let slot = leaf_count.checked_add(u32::try_from(i).unwrap()).unwrap();
        nodemap |= 1u32.checked_shl(slot).unwrap_or(0);
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(b"MTHN");
    buf.push(0x01); // codec version
    buf.extend_from_slice(&datamap.to_le_bytes());
    buf.extend_from_slice(&nodemap.to_le_bytes());
    buf.extend_from_slice(&leaf_count.to_le_bytes());
    buf.extend_from_slice(&child_count.to_le_bytes());
    for (et, sk, eid) in leaves {
        let key_json = hamt_leaf_key_json(et, sk);
        buf.extend_from_slice(&encode_lps(&key_json));
        buf.extend_from_slice(&encode_lps(eid));
    }
    for hash in child_hashes {
        buf.extend_from_slice(hash);
    }
    buf
}

#[test]
fn hamt_single_leaf_no_children() {
    let node = build_hamt_node(&[("m.room.name", "", "$abc:server")], &[]);
    let out = decode_hamt_node(&node).expect("should decode");
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("m.room.name"));
    assert!(text.contains("$abc:server"));
    assert!(text.contains("1 leaves"));
    assert!(text.contains("0 children"));
}

#[test]
fn hamt_multiple_leaves_with_children() {
    let h1 = [0x42u8; 32];
    let h2 = [0x99u8; 32];
    let node = build_hamt_node(
        &[
            ("m.room.member", "@a:b", "$ev1:server"),
            ("m.room.create", "", "$ev2:server"),
        ],
        &[h1, h2],
    );
    let out = decode_hamt_node(&node).expect("should decode");
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("@a:b"));
    assert!(text.contains("$ev1:server"));
    assert!(text.contains("$ev2:server"));
    assert!(text.contains(&hex::encode(&h1[..8])));
    assert!(text.contains(&hex::encode(&h2[..8])));
}

#[test]
fn hamt_empty_strings() {
    let node = build_hamt_node(&[("", "", "")], &[]);
    let out = decode_hamt_node(&node).expect("should decode");
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("leaf[0]"));
}

#[test]
fn hamt_empty_node() {
    let node = build_hamt_node(&[], &[]);
    let out = decode_hamt_node(&node).expect("should decode");
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("0 leaves"));
    assert!(text.contains("0 children"));
}

#[test]
fn hamt_state_group_root() {
    let room_prefix = [0xB2, 0xA2, 0xFD, 0xEA, 0xB7, 0x14, 0xDE, 0x6D];
    let room_id = "owusNddwskNpuHuitQ:test";
    let root_hash = [0xAB; 32];
    let lattice = [0xCD; 2048];
    let mut encoded = b"MTHR\x01".to_vec();
    encoded.extend_from_slice(&u16::try_from(room_prefix.len()).unwrap().to_be_bytes());
    encoded.extend_from_slice(&room_prefix);
    encoded.extend_from_slice(&u16::try_from(room_id.len()).unwrap().to_be_bytes());
    encoded.extend_from_slice(room_id.as_bytes());
    encoded.extend_from_slice(&root_hash);
    encoded.extend_from_slice(&lattice);
    assert_eq!(encoded.len(), 2120);

    let out = decode_hamt_root(&encoded).expect("should decode root");
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("b2a2fdeab714de6d"));
    assert!(text.contains(room_id));
    assert!(text.contains("abababababababab"));
    assert!(text.contains("2048 bytes"));
    assert!(text.contains("logical digest (BLAKE3): "));
}

#[test]
fn hamt_invalid_utf8_rejected() {
    let mut node = build_hamt_node(&[("m.room.name", "", "$ok")], &[]);
    // Corrupt the event type string: write an invalid UTF-8 sequence.
    // The length prefix says 10 bytes, but we fill with 0xFF.
    let invalid = b"\x0a\x00\x00\x00\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff";
    // Find where the first leaf's key starts (after the 21-byte header).
    node[21..21 + invalid.len()].copy_from_slice(invalid);
    assert!(decode_hamt_node(&node).is_none());
}

#[test]
fn hamt_wrong_version_rejected() {
    let mut node = build_hamt_node(&[("m.room.name", "", "$ok")], &[]);
    node[4] = 0x00;
    assert!(decode_hamt_node(&node).is_none());
}

#[test]
fn hamt_random_binary_not_hamt() {
    // Arbitrary payload starting with 0x01 should not decode merely from
    // the version byte; the complete header and body must also validate.
    let data = vec![0x01, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff];
    assert!(decode_hamt_node(&data).is_none());
}

#[test]
fn hamt_truncated_payload_rejected() {
    let full = build_hamt_node(&[("m.room.name", "", "$ok")], &[]);
    // Truncate after header -- leaf data missing.
    let trunc: Vec<u8> = full[..20].to_vec();
    assert!(decode_hamt_node(&trunc).is_none());
}

#[test]
fn hamt_overlong_payload_rejected() {
    let mut node = build_hamt_node(&[("m.room.name", "", "$ok")], &[]);
    node.extend_from_slice(&[0u8; 16]);
    assert!(decode_hamt_node(&node).is_none());
}

#[test]
fn hamt_pretty_print_returns_hamt() {
    let node = build_hamt_node(&[("m.room.topic", "room", "$t:server")], &[]);
    let out = pretty_print_payload(&node).expect("should recognize HAMT");
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("HAMT CHAMP"));
}

#[test]
fn hamt_datamap_nodemap_overlap_rejected() {
    // Both bitmaps claim slot 0.
    let mut buf = b"MTHN\x01".to_vec();
    buf.extend_from_slice(&1u32.to_le_bytes()); // datamap: slot 0
    buf.extend_from_slice(&1u32.to_le_bytes()); // nodemap: slot 0 (overlap)
    buf.extend_from_slice(&1u32.to_le_bytes()); // leaf_count
    buf.extend_from_slice(&1u32.to_le_bytes()); // child_count
    assert!(decode_hamt_node(&buf).is_none());
}

#[test]
fn hamt_leaf_count_mismatch_rejected() {
    // datamap says 1 leaf, but leaf_count says 0.
    let mut buf = b"MTHN\x01".to_vec();
    buf.extend_from_slice(&1u32.to_le_bytes()); // datamap
    buf.extend_from_slice(&0u32.to_le_bytes()); // nodemap
    buf.extend_from_slice(&0u32.to_le_bytes()); // leaf_count (wrong)
    buf.extend_from_slice(&0u32.to_le_bytes()); // child_count
    assert!(decode_hamt_node(&buf).is_none());
}

#[test]
fn hamt_rezzy_encoded_fixture() {
    use rezzy::hamt::PersistedInternalNode;

    // Build a node using rezzy's own encoder -- this is the ground
    // truth. K = String (JSON key), V = String (event_id), matching
    // the actual `HamtNode<String, String>` instantiation Synapse uses.
    let leaves = vec![
        (
            hamt_leaf_key_json("m.room.member", "@alice:example.org"),
            "$ev1:example.org".to_owned(),
        ),
        (
            hamt_leaf_key_json("m.room.name", ""),
            "$ev2:example.org".to_owned(),
        ),
    ];
    let child_hash = [0xABu8; 32];
    let node = PersistedInternalNode {
        datamap: 0b0011, // slots 0 and 1
        nodemap: 0b0100, // slot 2
        leaves,
        child_hashes: vec![child_hash],
    };
    let encoded = node.encode_v1();

    // Our decoder must accept these exact bytes.
    let out = decode_hamt_node(&encoded).expect("rezzy-encoded node must decode");
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("@alice:example.org"));
    assert!(text.contains("$ev1:example.org"));
    assert!(text.contains("$ev2:example.org"));
    assert!(text.contains("m.room.member"));
    assert!(text.contains("m.room.name"));
    assert!(text.contains(&hex::encode(&child_hash[..8])));
}

#[test]
fn hamt_rezzy_roundtrip_single_leaf() {
    use rezzy::hamt::PersistedInternalNode;

    let node = PersistedInternalNode {
        datamap: 0x01,
        nodemap: 0x00,
        leaves: vec![(
            hamt_leaf_key_json("m.room.create", ""),
            "$create:server".to_owned(),
        )],
        child_hashes: vec![],
    };
    let encoded = node.encode_v1();
    let out = decode_hamt_node(&encoded).expect("must decode");
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("m.room.create"));
    assert!(text.contains("$create:server"));
    assert!(text.contains("1 leaves"));
    assert!(text.contains("0 children"));
}

// ── Import → event DAG → auth chain → state group integration tests ──

#[test]
fn build_event_dag_resolves_prev_and_auth_edges() {
    let create = owned_value(
        r#"{"event_id":"$create","room_id":"!r:x","type":"m.room.create","state_key":"","sender":"@a:x","content":{"creator":"@a:x"}}"#,
    );
    let member = owned_value(
        r#"{"event_id":"$join","room_id":"!r:x","type":"m.room.member","state_key":"@a:x","sender":"@a:x","prev_events":["$create"],"auth_events":["$create"],"content":{"membership":"join"}}"#,
    );
    let msg = owned_value(
        r#"{"event_id":"$msg","room_id":"!r:x","type":"m.room.message","sender":"@a:x","prev_events":["$join"],"auth_events":["$join","$create"],"content":{"body":"hi"}}"#,
    );
    let (frontier, _id_map, _rev) = build_event_dag(&[create, member, msg]);
    assert_eq!(frontier.nodes.len(), 3);
    // $msg should have 2 prev edges and 2 auth edges
    let msg_node = &frontier.nodes[2];
    assert_eq!(msg_node.prev.1, 1); // 1 prev_events
    assert_eq!(msg_node.auth.1, 2); // 2 auth_events
}

/// A database root and its (lazily created) pools, for the adjacency tests.
fn adjacency_db(name: &str) -> (Database, PathBuf) {
    let root = unique_temp_dir().join(name);
    (Database::open(root.clone()).unwrap(), root)
}

fn room_adjacency(room: &str) -> mtxdb::csr_adjacency::CsrAdjacency {
    mtxdb::csr_adjacency::CsrAdjacency::new(room)
}

#[test]
fn csr_adjacency_records_prev_auth_and_relation() {
    let (db, root) = adjacency_db("adjacency_basic");
    let template = default_matrix_import_template();
    let event = owned_value(
        r#"{"event_id":"$source","room_id":"!r:x","type":"m.room.message","prev_events":["$prev1",["$prev2",{}]],"auth_events":["$auth"],"content":{"m.relates_to":{"rel_type":"m.reference","event_id":"$related"}}}"#,
    );
    record_csr_adjacency(&db, &template, std::slice::from_ref(&event)).unwrap();

    let adjacency = room_adjacency("!r:x");
    assert_eq!(
        adjacency.prev_of(&db, "$source").unwrap(),
        Some(vec!["$prev1".to_owned(), "$prev2".to_owned()]),
        "both the plain and the [id, hashes] reference forms are recorded"
    );
    assert_eq!(
        adjacency.auth_of(&db, "$source").unwrap(),
        Some(vec!["$auth".to_owned()])
    );
    let relation = adjacency
        .relation_of(&db, "$source", &mtxdb::csr_adjacency::AlwaysVisible)
        .unwrap()
        .expect("the relation is recorded");
    assert_eq!(
        (relation.target.as_str(), relation.rel_type.as_str()),
        ("$related", "m.reference")
    );

    // Recording again is a no-op.
    record_csr_adjacency(&db, &template, &[event]).unwrap();
    assert!(adjacency.verify(&db).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn csr_adjacency_keeps_leaf_events_distinct_from_referenced_ones() {
    let (db, root) = adjacency_db("adjacency_leaf");
    let template = default_matrix_import_template();
    let leaf = owned_value(
        r#"{"event_id":"$leaf","room_id":"!r:x","type":"m.room.message","content":{"body":"leaf"}}"#,
    );
    let child = owned_value(
        r#"{"event_id":"$child","room_id":"!r:x","type":"m.room.message","prev_events":["$ghost"],"auth_events":["$leaf"]}"#,
    );
    record_csr_adjacency(&db, &template, &[leaf, child]).unwrap();

    let adjacency = room_adjacency("!r:x");
    assert_eq!(adjacency.auth_of(&db, "$leaf").unwrap(), Some(Vec::new()));
    assert_eq!(adjacency.prev_of(&db, "$leaf").unwrap(), Some(Vec::new()));
    assert_eq!(
        adjacency.auth_of(&db, "$ghost").unwrap(),
        None,
        "an event only referenced by another has no recorded adjacency"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn csr_adjacency_groups_duplicates_and_separates_rooms() {
    let (db, root) = adjacency_db("adjacency_rooms");
    let template = default_matrix_import_template();
    let event_a = owned_value(r#"{"event_id":"$same","room_id":"!a:x","prev_events":["$p"]}"#);
    let event_a_duplicate =
        owned_value(r#"{"event_id":"$same","room_id":"!a:x","prev_events":["$p"]}"#);
    let event_b = owned_value(r#"{"event_id":"$same","room_id":"!b:x","prev_events":["$q"]}"#);
    record_csr_adjacency(&db, &template, &[event_a, event_a_duplicate, event_b]).unwrap();

    assert_eq!(
        room_adjacency("!a:x").prev_of(&db, "$same").unwrap(),
        Some(vec!["$p".to_owned()])
    );
    assert_eq!(
        room_adjacency("!b:x").prev_of(&db, "$same").unwrap(),
        Some(vec!["$q".to_owned()])
    );
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn csr_adjacency_rejects_a_conflict_without_recording_any_of_the_batch() {
    let (db, root) = adjacency_db("adjacency_conflict");
    let template = default_matrix_import_template();
    let adjacency = room_adjacency("!a:x");
    let original = owned_value(
        r#"{"event_id":"$same","room_id":"!a:x","prev_events":["$p"],"auth_events":["$auth"]}"#,
    );
    record_csr_adjacency(&db, &template, std::slice::from_ref(&original)).unwrap();

    // A later import whose event differs from the recorded one fails as a unit:
    // the good event before it is not recorded either.
    let fine = owned_value(r#"{"event_id":"$fine","room_id":"!a:x","prev_events":["$p"]}"#);
    let conflicting = owned_value(
        r#"{"event_id":"$same","room_id":"!a:x","prev_events":["$other"],"auth_events":["$auth"]}"#,
    );
    let error = record_csr_adjacency(&db, &template, &[fine, conflicting])
        .expect_err("a differing prev edge must be rejected");
    assert!(
        format!("{error:#}").contains("recording auth adjacency for room !a:x"),
        "saw: {error:#}"
    );
    assert_eq!(adjacency.short_id(&db, "$fine").unwrap(), None);
    assert_eq!(
        adjacency.prev_of(&db, "$same").unwrap(),
        Some(vec!["$p".to_owned()]),
        "the recorded adjacency is untouched"
    );

    // Two conflicting copies inside one batch fail the same way.
    let first = owned_value(r#"{"event_id":"$dup","room_id":"!a:x","prev_events":["$p"]}"#);
    let second = owned_value(r#"{"event_id":"$dup","room_id":"!a:x","prev_events":["$q"]}"#);
    assert!(record_csr_adjacency(&db, &template, &[first, second]).is_err());
    assert_eq!(adjacency.short_id(&db, "$dup").unwrap(), None);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn csr_adjacency_keeps_a_relation_across_redacted_copies() {
    let (db, root) = adjacency_db("adjacency_redaction");
    let template = default_matrix_import_template();
    let adjacency = room_adjacency("!a:x");
    let visible = mtxdb::csr_adjacency::AlwaysVisible;
    // Redaction keeps event_id, prev_events and auth_events but empties content.
    let unredacted = owned_value(
        r#"{"event_id":"$same","room_id":"!a:x","prev_events":["$p"],"auth_events":["$auth"],"content":{"m.relates_to":{"rel_type":"m.reference","event_id":"$target"}}}"#,
    );
    let redacted = owned_value(
        r#"{"event_id":"$same","room_id":"!a:x","prev_events":["$p"],"auth_events":["$auth"],"content":{}}"#,
    );
    let target = |db: &Database| {
        adjacency
            .relation_of(db, "$same", &visible)
            .unwrap()
            .map(|relation| relation.target)
    };

    record_csr_adjacency(&db, &template, std::slice::from_ref(&unredacted)).unwrap();
    record_csr_adjacency(&db, &template, std::slice::from_ref(&redacted))
        .expect("a redacted re-import must not be rejected");
    assert_eq!(
        target(&db).as_deref(),
        Some("$target"),
        "the stored relation survives a redacted re-import"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(root);

    // Seen redacted first, then unredacted in the same input: the relation is added.
    let (db, root) = adjacency_db("adjacency_redaction_order");
    record_csr_adjacency(&db, &template, &[redacted, unredacted]).unwrap();
    assert_eq!(target(&db).as_deref(), Some("$target"));
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn csr_adjacency_ignores_reply_fallbacks_and_overlong_relation_types() {
    let (db, root) = adjacency_db("adjacency_relation_edges");
    let template = default_matrix_import_template();
    let adjacency = room_adjacency("!a:x");
    let visible = mtxdb::csr_adjacency::AlwaysVisible;
    // A rich-reply fallback has no `rel_type`, so it is not a relation.
    let reply = owned_value(
        r#"{"event_id":"$reply","room_id":"!a:x","prev_events":["$p"],"content":{"m.relates_to":{"m.in_reply_to":{"event_id":"$target"}}}}"#,
    );
    // A `rel_type` too long to be a real identifier is dropped, not an error.
    let long_rel_type = "x".repeat(300);
    let long = owned_value(&format!(
        r#"{{"event_id":"$long","room_id":"!a:x","prev_events":["$p"],"content":{{"m.relates_to":{{"rel_type":"{long_rel_type}","event_id":"$target"}}}}}}"#
    ));
    record_csr_adjacency(&db, &template, &[reply, long])
        .expect("an unrepresentable relation must not fail the import");

    for event in ["$reply", "$long"] {
        assert!(adjacency
            .relation_of(&db, event, &visible)
            .unwrap()
            .is_none());
        assert_eq!(
            adjacency.prev_of(&db, event).unwrap(),
            Some(vec!["$p".to_owned()]),
            "the event's own adjacency is still recorded"
        );
    }
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn csr_adjacency_is_not_recorded_for_templates_without_event_id_identity() {
    let (db, root) = adjacency_db("adjacency_other_identity");
    // A record identity other than /event_id cannot name a referenced event.
    let template = sender_identity_template();
    let event =
        owned_value(r#"{"event_id":"$e","room_id":"!r:x","sender":"@alice","prev_events":["$p"]}"#);
    record_csr_adjacency(&db, &template, &[event]).unwrap();
    assert_eq!(room_adjacency("!r:x").short_id(&db, "$e").unwrap(), None);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn csr_adjacency_skips_events_without_a_room_or_event_id() {
    let (db, root) = adjacency_db("adjacency_skips");
    let template = default_matrix_import_template();
    let no_room = owned_value(r#"{"event_id":"$orphan","prev_events":["$p"]}"#);
    let no_id = owned_value(r#"{"room_id":"!r:x","prev_events":["$p"]}"#);
    let fine = owned_value(r#"{"event_id":"$fine","room_id":"!r:x"}"#);
    record_csr_adjacency(&db, &template, &[no_room, no_id, fine]).unwrap();
    let adjacency = room_adjacency("!r:x");
    assert!(adjacency.short_id(&db, "$fine").unwrap().is_some());
    assert_eq!(adjacency.short_id(&db, "$orphan").unwrap(), None);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn csr_adjacency_splits_a_large_room_into_batches() {
    let (db, root) = adjacency_db("adjacency_large");
    let template = default_matrix_import_template();
    // One more than a batch holds, so the room is recorded in two transactions.
    let count = mtxdb::room_auth::MAX_BATCH_EVENTS + 1;
    let events: Vec<OwnedValue> = (0..count)
        .map(|index| {
            let auth = if index == 0 {
                String::new()
            } else {
                format!(r#""$e{}""#, index - 1)
            };
            owned_value(&format!(
                r#"{{"event_id":"$e{index}","room_id":"!big:x","auth_events":[{auth}]}}"#
            ))
        })
        .collect();
    record_csr_adjacency(&db, &template, &events).unwrap();

    let auth = RoomAuth::new("!big:x");
    let last = format!("$e{}", count - 1);
    assert_eq!(
        auth.auth_chain(&db, &last).unwrap().len(),
        count - 1,
        "every event of both batches is recorded and chained"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn an_import_records_auth_adjacency_that_survives_a_reopen() {
    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let input = root.join("export.json");
    std::fs::write(
        &input,
        r#"{
            "pdus": [
                {"event_id":"$c","room_id":"!room","sender":"@server",
                 "type":"m.room.create","state_key":"",
                 "content":{"creator":"@server","room_version":"10"},
                 "auth_events":[],"prev_events":[]},
                {"event_id":"$m","room_id":"!room","sender":"@server",
                 "type":"m.room.message","content":{},
                 "auth_events":["$c","$member"],"prev_events":["$c"]},
                {"event_id":"$x","room_id":"!room","sender":"@server",
                 "type":"m.room.message","content":{},
                 "auth_events":["$c","$member","$m"],"prev_events":["$m"]}
            ],
            "auth_chain": [
                {"event_id":"$member","room_id":"!room","sender":"@server",
                 "type":"m.room.member","state_key":"@server",
                 "content":{"membership":"join"},"auth_events":["$c"]}
            ]
        }"#,
    )
    .unwrap();
    let template = default_matrix_import_template();
    let mut established = HashSet::new();
    cmd_import_file(
        &db,
        db.event_dag(),
        db.event_dag(),
        &input,
        None,
        &template,
        &mut established,
        false,
        None,
        true,
    )
    .unwrap();
    db.state().sync_all().unwrap();
    drop(db);

    let db = Database::open(root.clone()).unwrap();
    let auth = RoomAuth::new("!room");
    let mut edges = auth.auth_edges(&db, "$x").unwrap();
    edges.sort();
    assert_eq!(
        edges,
        vec!["$c", "$m", "$member"],
        "order is not part of the contract"
    );
    let mut chain = auth.auth_chain(&db, "$x").unwrap();
    chain.sort();
    assert_eq!(chain, vec!["$c", "$m", "$member"]);
    // The auth-chain event was recorded through the same path as the PDUs.
    assert_eq!(auth.auth_edges(&db, "$member").unwrap(), vec!["$c"]);
    assert_eq!(auth.auth_edges(&db, "$c").unwrap(), Vec::<String>::new());
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn compute_state_groups_inherit_and_split_on_derivation() {
    let mut events = Vec::new();
    events.push(owned_value(
        r#"{"event_id":"$create","room_id":"!r:x","type":"m.room.create","state_key":"","sender":"@a:x","content":{}}"#,
    ));
    let mut prev = "$create".to_owned();
    // A long chain: every 50th event is a state event, the rest messages.
    for i in 0..5000u32 {
        let id = format!("$e{i}");
        let json = if i % 50 == 0 {
            format!(
                r#"{{"event_id":"{id}","room_id":"!r:x","type":"m.room.member","state_key":"@u{i}:x","sender":"@a:x","prev_events":["{prev}"],"content":{{}}}}"#
            )
        } else {
            format!(
                r#"{{"event_id":"{id}","room_id":"!r:x","type":"m.room.message","sender":"@a:x","prev_events":["{prev}"],"content":{{}}}}"#
            )
        };
        events.push(owned_value(&json));
        prev = id;
    }
    // Fork off the tip: two branches set the same key differently, then a
    // merge event lists both parents.
    events.push(owned_value(&format!(
        r#"{{"event_id":"$left","room_id":"!r:x","type":"m.room.topic","state_key":"","sender":"@a:x","prev_events":["{prev}"],"content":{{}}}}"#
    )));
    events.push(owned_value(&format!(
        r#"{{"event_id":"$right","room_id":"!r:x","type":"m.room.topic","state_key":"","sender":"@a:x","prev_events":["{prev}"],"content":{{}}}}"#
    )));
    events.push(owned_value(
        r#"{"event_id":"$merge","room_id":"!r:x","type":"m.room.message","sender":"@a:x","prev_events":["$left","$right"],"content":{}}"#,
    ));

    let groups = compute_state_groups_partial(&events, &[], "!r:x").groups;
    assert_eq!(groups.len(), events.len());
    // Messages between two state events share one instance; a state event
    // starts a new derivation.
    assert_eq!(groups["$e1"], groups["$e49"]);
    assert_ne!(groups["$e49"], groups["$e50"]);
    assert_ne!(groups["$create"], groups["$e0"]);
    // The forks differ, and the merge is its own derivation. Its fixture
    // declares no room version, so it falls back to the first-parent rule and
    // the resolved state matches the first parent.
    assert_ne!(groups["$left"], groups["$right"]);
    assert_ne!(groups["$merge"], groups["$left"]);
    assert_ne!(groups["$merge"], groups["$right"]);
}

#[test]
fn rejected_events_never_contribute_state_and_soft_failed_only_in_the_client_view() {
    let groups = |flag: &str, view: super::StateView| {
        let create = owned_value(
            r#"{"event_id":"$create","room_id":"!r:x","type":"m.room.create","state_key":"","content":{}}"#,
        );
        let flagged = owned_value(&format!(
            r#"{{"event_id":"$bad","room_id":"!r:x","type":"m.room.name","state_key":"","prev_events":["$create"],{flag}"content":{{"name":"x"}}}}"#
        ));
        let after = owned_value(
            r#"{"event_id":"$after","room_id":"!r:x","type":"m.room.message","prev_events":["$bad"],"content":{}}"#,
        );
        super::compute_state_groups_partial_in_view(&[create, flagged, after], &[], view, "!r:x")
            .groups
    };
    let federated = super::StateView::Federated;
    let client = super::StateView::Client;
    let accepted = groups("", federated);
    assert_ne!(accepted["$create"], accepted["$bad"]);

    for key in ["__rejected", "_rejected", "rejected"] {
        for view in [federated, client] {
            let g = groups(&format!(r#""{key}":true,"#), view);
            assert_eq!(g["$create"], g["$bad"], "{key} {view:?}");
            assert_eq!(g["$bad"], g["$after"], "{key} {view:?}");
        }
    }
    for key in [
        "__soft-failed",
        "_soft-failed",
        "soft-failed",
        "soft_failed",
        "__soft_failed",
    ] {
        let flag = format!(r#""{key}":true,"#);
        // Federated state keeps the soft-failed event...
        let fed = groups(&flag, federated);
        assert_ne!(fed["$create"], fed["$bad"], "{key} federated");
        assert_eq!(fed["$bad"], fed["$after"], "{key} federated");
        assert_eq!(fed["$after"], accepted["$after"], "{key} federated");
        // ...the client view drops it, so the groups differ.
        let cli = groups(&flag, client);
        assert_eq!(cli["$create"], cli["$bad"], "{key} client");
        assert_ne!(fed["$after"], cli["$after"], "{key}");
    }
    // `false` is not a flag.
    let not_flagged = groups(r#""__rejected":false,"#, federated);
    assert_ne!(not_flagged["$create"], not_flagged["$bad"]);
}

#[test]
fn compute_state_groups_assigns_groups_per_event() {
    let create = owned_value(
        r#"{"event_id":"$create","room_id":"!r:x","type":"m.room.create","state_key":"","sender":"@a:x","content":{"creator":"@a:x"}}"#,
    );
    let member = owned_value(
        r#"{"event_id":"$join","room_id":"!r:x","type":"m.room.member","state_key":"@a:x","sender":"@a:x","prev_events":["$create"],"auth_events":["$create"],"content":{"membership":"join"}}"#,
    );
    let msg = owned_value(
        r#"{"event_id":"$msg","room_id":"!r:x","type":"m.room.message","sender":"@a:x","prev_events":["$join"],"auth_events":["$join"],"content":{"body":"hi"}}"#,
    );
    let groups = compute_state_groups_partial(&[create, member, msg], &[], "!r:x").groups;
    // All three events should have a state-group instance.
    assert!(groups.contains_key("$create"));
    assert!(groups.contains_key("$join"));
    assert!(groups.contains_key("$msg"));
    for sg in groups.values() {
        assert!(valid_state_group_id(sg));
    }
    // The message inherits its parent's instance; the state event starts a new
    // derivation.
    assert_ne!(groups["$create"], groups["$join"]);
    assert_eq!(groups["$join"], groups["$msg"]);
}

#[test]
fn compute_state_groups_empty_input() {
    let groups = compute_state_groups_partial(&[], &[], "!r:x").groups;
    assert!(groups.is_empty());
}

#[test]
fn state_group_id_validation_requires_a_128_bit_instance_id() {
    let golden = [b'A'; STATE_GROUP_ID_LENGTH];
    assert!(valid_state_group_id(&golden));
    assert!(!valid_state_group_id(&[0u8; STATE_GROUP_ID_LENGTH - 1]));
    assert!(!valid_state_group_id(&[0u8; STATE_GROUP_ID_LENGTH + 1]));
    assert!(!valid_state_group_id(&[]));
}

#[test]
fn state_group_cache_loads_complete_values_and_falls_back_on_missing_or_invalid() {
    let dir = unique_temp_dir().join("state_group_cache");
    std::fs::create_dir_all(&dir).unwrap();
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let aux = mtxdb::auxiliary::AuxiliaryIndex::open(&store, STATE_GROUP_NAMESPACE);
    aux.ensure_metadata().unwrap();
    let first = [b'A'; STATE_GROUP_ID_LENGTH];
    let second = [b'B'; STATE_GROUP_ID_LENGTH];
    aux.put_many(&[
        (b"first".as_slice(), first.as_slice()),
        (b"second".as_slice(), second.as_slice()),
    ])
    .unwrap();
    let ids = vec!["first".to_owned(), "second".to_owned()];
    let loaded = load_state_groups(&aux, &ids).unwrap();
    let StateGroupLoad::Complete(loaded) = loaded else {
        panic!("complete cache should load");
    };
    assert_eq!(loaded["first"], first);
    assert_eq!(loaded["second"], second);

    let missing = vec!["first".to_owned(), "absent".to_owned()];
    assert!(matches!(
        load_state_groups(&aux, &missing).unwrap(),
        StateGroupLoad::Missing
    ));
    aux.put(b"invalid", &[0xff]).unwrap();
    let invalid = vec!["invalid".to_owned()];
    assert!(matches!(
        load_state_groups(&aux, &invalid).unwrap(),
        StateGroupLoad::Invalid { .. }
    ));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn state_groups_from_values_treats_a_cardinality_mismatch_as_invalid() {
    let ids = vec!["first".to_owned(), "second".to_owned()];
    let value = Some(vec![b'A'; STATE_GROUP_ID_LENGTH]);
    assert!(matches!(
        super::state_groups_from_values(&ids, vec![value.clone()]),
        StateGroupLoad::Invalid { .. }
    ));
    assert!(matches!(
        super::state_groups_from_values(&["only".to_owned()], vec![None, None]),
        StateGroupLoad::Invalid { .. }
    ));
    assert!(matches!(
        super::state_groups_from_values(&ids, vec![value.clone(), None]),
        StateGroupLoad::Missing
    ));
    let StateGroupLoad::Complete(groups) =
        super::state_groups_from_values(&ids, vec![value.clone(), value])
    else {
        panic!("matching cardinality with valid values should load");
    };
    assert_eq!(groups.len(), 2);
}

#[test]
fn partial_state_groups_repair_when_a_missing_parent_arrives() {
    let dir = unique_temp_dir().join("state_group_repair");
    std::fs::create_dir_all(&dir).unwrap();
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let aux = mtxdb::auxiliary::AuxiliaryIndex::open(&store, STATE_GROUP_NAMESPACE);
    aux.ensure_metadata().unwrap();
    let incomplete = owned_value(
        r#"{"event_id":"$child","room_id":"!r:x","type":"m.room.message","prev_events":["$parent"],"content":{}}"#,
    );
    let first = compute_state_groups_partial(std::slice::from_ref(&incomplete), &[], "!r:x");
    assert!(!first.groups.contains_key("$child"));
    assert_ne!(first.unresolved, Vec::<String>::new());
    assert_eq!(aux.get(b"$child").unwrap(), None);

    let parent = owned_value(
        r#"{"event_id":"$parent","room_id":"!r:x","type":"m.room.create","content":{}}"#,
    );
    let repaired = compute_state_groups_partial(&[parent, incomplete], &[], "!r:x");
    assert!(repaired.groups.contains_key("$child"));
    assert_eq!(repaired.unresolved, Vec::<String>::new());
    let entries: Vec<(&[u8], &[u8])> = repaired
        .groups
        .iter()
        .map(|(event_id, instance_id)| (event_id.as_bytes(), instance_id.as_slice()))
        .collect();
    aux.put_many(&entries).unwrap();
    let repaired_id = aux.get(b"$child").unwrap().unwrap();
    assert_eq!(repaired_id.len(), STATE_GROUP_ID_LENGTH);
    assert!(valid_state_group_id(&repaired_id));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn import_ordering_ignores_edges_to_absent_events() {
    let known = owned_value(
        r#"{"event_id":"$known","room_id":"!r:x","type":"m.room.create","content":{}}"#,
    );
    let child = owned_value(
        r#"{"event_id":"$ordering-child","room_id":"!r:x","type":"m.room.message","prev_events":["$known","$absent"],"content":{}}"#,
    );
    let order = topological_event_order(&[child, known]).expect("in-batch parent");
    assert!(order["$known"] < order["$ordering-child"]);
}

#[test]
fn partial_state_groups_report_cycles() {
    let first = owned_value(
        r#"{"event_id":"$first","room_id":"!r:x","type":"m.room.message","prev_events":["$second"],"content":{}}"#,
    );
    let second = owned_value(
        r#"{"event_id":"$second","room_id":"!r:x","type":"m.room.message","prev_events":["$first"],"content":{}}"#,
    );
    let result = compute_state_groups_partial(&[first, second], &[], "!r:x");
    assert!(result.groups.is_empty());
    assert_eq!(result.unresolved.len(), 2);
}

#[test]
fn compute_state_groups_deterministic() {
    let create = owned_value(
        r#"{"event_id":"$create","room_id":"!r:x","type":"m.room.create","state_key":"","sender":"@a:x","content":{}}"#,
    );
    let member = owned_value(
        r#"{"event_id":"$join","room_id":"!r:x","type":"m.room.member","state_key":"@a:x","sender":"@a:x","prev_events":["$create"],"auth_events":["$create"],"content":{}}"#,
    );
    let g1 = compute_state_groups_partial(&[create.clone(), member.clone()], &[], "!r:x").groups;
    let g2 = compute_state_groups_partial(&[create, member], &[], "!r:x").groups;
    assert_eq!(g1, g2);
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one fixture exercises several batch-composition invariants together"
)]
fn compute_state_groups_is_stable_across_overlapping_batches() {
    let create = owned_value(
        r#"{"event_id":"$create","room_id":"!r:x","type":"m.room.create","state_key":"","sender":"@a:x","content":{}}"#,
    );
    let member = owned_value(
        r#"{"event_id":"$join","room_id":"!r:x","type":"m.room.member","state_key":"@a:x","sender":"@a:x","prev_events":["$create"],"auth_events":["$create"],"content":{}}"#,
    );
    let sibling = owned_value(
        r#"{"event_id":"$sibling","room_id":"!r:x","type":"m.room.topic","state_key":"","sender":"@a:x","prev_events":["$create"],"content":{}}"#,
    );
    let branch_a = owned_value(
        r#"{"event_id":"$branch-a","room_id":"!r:x","type":"m.room.topic","state_key":"","sender":"@a:x","prev_events":["$create"],"content":{}}"#,
    );
    let branch_b = owned_value(
        r#"{"event_id":"$branch-b","room_id":"!r:x","type":"m.room.name","state_key":"","sender":"@a:x","prev_events":["$create"],"content":{}}"#,
    );
    let merge = owned_value(
        r#"{"event_id":"$merge","room_id":"!r:x","type":"m.room.message","sender":"@a:x","prev_events":["$branch-a","$branch-b"],"content":{}}"#,
    );
    let orphan = owned_value(
        r#"{"event_id":"$orphan","room_id":"!r:x","type":"m.room.message","sender":"@a:x","prev_events":["$missing"],"content":{}}"#,
    );

    let base = compute_state_groups_partial(
        &[
            create.clone(),
            member.clone(),
            branch_a.clone(),
            branch_b.clone(),
            merge.clone(),
        ],
        &[],
        "!r:x",
    );
    let with_fork = compute_state_groups_partial(
        &[
            create.clone(),
            member.clone(),
            branch_a.clone(),
            branch_b.clone(),
            merge.clone(),
            sibling.clone(),
        ],
        &[],
        "!r:x",
    );
    let with_orphan = compute_state_groups_partial(
        &[
            create.clone(),
            member.clone(),
            branch_a.clone(),
            branch_b.clone(),
            merge.clone(),
            orphan,
        ],
        &[],
        "!r:x",
    );
    let with_auth_chain = compute_state_groups_partial(
        &[
            create.clone(),
            member.clone(),
            branch_a.clone(),
            branch_b.clone(),
            merge.clone(),
        ],
        &[sibling],
        "!r:x",
    );
    let reordered = compute_state_groups_partial(
        &[
            merge.clone(),
            branch_b.clone(),
            branch_a.clone(),
            member.clone(),
            create.clone(),
        ],
        &[],
        "!r:x",
    );
    let without_branch_b = compute_state_groups_partial(
        &[
            create.clone(),
            member.clone(),
            branch_a.clone(),
            merge.clone(),
        ],
        &[],
        "!r:x",
    );

    assert_eq!(base.groups["$join"], with_fork.groups["$join"]);
    assert_eq!(base.groups["$join"], with_orphan.groups["$join"]);
    assert_eq!(base.groups["$join"], with_auth_chain.groups["$join"]);
    assert_eq!(base.groups["$merge"], with_fork.groups["$merge"]);
    assert_eq!(base.groups["$merge"], with_orphan.groups["$merge"]);
    assert_eq!(base.groups["$merge"], with_auth_chain.groups["$merge"]);
    assert_eq!(base.groups["$merge"], reordered.groups["$merge"]);
    assert_ne!(base.groups["$merge"], base.groups["$branch-a"]);
    assert_ne!(base.groups["$merge"], base.groups["$branch-b"]);
    assert!(without_branch_b.groups.contains_key("$branch-a"));
    assert!(!without_branch_b.groups.contains_key("$merge"));
    assert!(without_branch_b.unresolved.iter().any(|id| id == "$merge"));
    assert!(!with_orphan.groups.contains_key("$orphan"));
    assert!(with_orphan.unresolved.iter().any(|id| id == "$orphan"));
}

#[test]
fn first_parent_fallback_when_no_room_version_is_declared() {
    let create = owned_value(
        r#"{"event_id":"$create","room_id":"!r:x","type":"m.room.create","state_key":"","sender":"@a:x","content":{}}"#,
    );
    let first = owned_value(
        r#"{"event_id":"$first","room_id":"!r:x","type":"m.room.topic","state_key":"","sender":"@a:x","prev_events":["$create"],"content":{}}"#,
    );
    let second = owned_value(
        r#"{"event_id":"$second","room_id":"!r:x","type":"m.room.topic","state_key":"","sender":"@a:x","prev_events":["$create"],"content":{}}"#,
    );
    let merge_first = owned_value(
        r#"{"event_id":"$merge-first","room_id":"!r:x","type":"m.room.message","sender":"@a:x","prev_events":["$first","$second"],"content":{}}"#,
    );
    let merge_second = owned_value(
        r#"{"event_id":"$merge-second","room_id":"!r:x","type":"m.room.message","sender":"@a:x","prev_events":["$second","$first"],"content":{}}"#,
    );

    // This batch declares no room version, so no state-resolution rules
    // apply and the merge falls back to the first-parent rule: the
    // same-key conflict is decided by the first prev_events entry, which
    // makes the two merge orders disagree. The merge still gets its own
    // instance because it has two distinct parents.
    let first_groups = compute_state_groups_partial(
        &[create.clone(), first.clone(), second.clone(), merge_first],
        &[],
        "!r:x",
    )
    .groups;
    let second_groups =
        compute_state_groups_partial(&[create, first, second, merge_second], &[], "!r:x").groups;

    assert_ne!(first_groups["$first"], first_groups["$second"]);
    assert_ne!(first_groups["$merge-first"], first_groups["$first"]);
    assert_ne!(first_groups["$merge-first"], first_groups["$second"]);
    assert_ne!(second_groups["$merge-second"], second_groups["$second"]);
    assert_ne!(second_groups["$merge-second"], second_groups["$first"]);
}

#[test]
fn verify_auth_chain_edges_detects_dangling() {
    let create = owned_value(
        r#"{"event_id":"$create","room_id":"!r:x","type":"m.room.create","sender":"@a:x","content":{}}"#,
    );
    let member = owned_value(
        r#"{"event_id":"$join","room_id":"!r:x","type":"m.room.member","sender":"@a:x","auth_events":["$create"],"content":{}}"#,
    );
    // All edges resolve.
    let dangling = verify_auth_chain_edges(&[create.clone(), member.clone()], &[]);
    assert_eq!(dangling, Vec::<(String, String)>::new());

    // $missing is not in either set.
    let bad = owned_value(
        r#"{"event_id":"$bad","room_id":"!r:x","type":"m.room.message","sender":"@a:x","auth_events":["$missing"],"content":{}}"#,
    );
    let dangling = verify_auth_chain_edges(&[create, member, bad], &[]);
    assert_eq!(dangling.len(), 1);
    assert_eq!(dangling[0].0, "$bad");
    assert_eq!(dangling[0].1, "$missing");
}

#[test]
fn verify_auth_chain_edges_auth_chain_as_known_set() {
    // An auth_events reference to an event only in the auth_chain
    // (not in pdus) should still resolve.
    let create = owned_value(
        r#"{"event_id":"$create","room_id":"!r:x","type":"m.room.create","sender":"@a:x","content":{}}"#,
    );
    let member = owned_value(
        r#"{"event_id":"$join","room_id":"!r:x","type":"m.room.member","sender":"@a:x","auth_events":["$create"],"content":{}}"#,
    );
    let dangling = verify_auth_chain_edges(&[member], &[create]);
    assert_eq!(dangling, Vec::<(String, String)>::new());
}

#[test]
fn parse_federation_input_separates_pdus_and_auth_chain() {
    let json = r#"{
        "pdus": [{"event_id": "$p1", "room_id": "!r:x", "type": "m.room.message"}],
        "auth_chain": [{"event_id": "$a1", "room_id": "!r:x", "type": "m.room.create"}]
    }"#;
    let input = parse_federation_input(json.as_bytes()).unwrap();
    assert_eq!(input.pdus.len(), 1);
    assert_eq!(input.auth_chain.len(), 1);
    assert_eq!(event_id(&input.pdus[0]), Some("$p1"));
    assert_eq!(event_id(&input.auth_chain[0]), Some("$a1"));
}

#[test]
fn parse_federation_input_empty_rejected() {
    let json = r#"{"unrelated": true}"#;
    assert!(parse_federation_input(json.as_bytes()).is_err());
}

#[test]
fn parse_federation_input_requires_an_array_and_rejects_wrong_types() {
    for json in [
        r#"{"unrelated": true}"#,
        r#"{"pdus": null, "auth_chain": []}"#,
        r#"{"pdus": [], "auth_chain": null}"#,
        r#"{"pdus": 3, "auth_chain": []}"#,
        r#"{"pdus": {}, "auth_chain": []}"#,
    ] {
        assert!(
            parse_federation_input(json.as_bytes()).is_err(),
            "malformed federation document was accepted: {json}"
        );
    }
}

#[test]
fn parse_federation_input_accepts_either_array_or_both() {
    for json in [
        r#"{"pdus": []}"#,
        r#"{"auth_chain": []}"#,
        r#"{"pdus": [], "auth_chain": []}"#,
    ] {
        let input = parse_federation_input(json.as_bytes()).unwrap();
        assert_eq!(input.pdus, Vec::<OwnedValue>::new());
        assert_eq!(input.auth_chain, Vec::<OwnedValue>::new());
    }
}

#[test]
fn state_set_digest_is_deterministic() {
    let mut s1 = StateSet::new();
    s1.set("m.room.create", "", "$create".into());
    s1.set("m.room.member", "@a:x", "$join".into());
    let mut s2 = StateSet::new();
    s2.set("m.room.create", "", "$create".into());
    s2.set("m.room.member", "@a:x", "$join".into());
    assert_eq!(s1.digest_base64url(), s2.digest_base64url());
}

#[test]
fn state_set_merge_takes_first_wins() {
    let mut base = StateSet::new();
    base.set("m.room.name", "", "$old".into());
    let mut override_ = StateSet::new();
    override_.set("m.room.name", "", "$new".into());
    override_.set("m.room.topic", "", "$topic".into());
    base.merge(&override_);
    // First-wins: $old should survive because base already had the key.
    assert_eq!(
        &base.entries[&("m.room.name".into(), String::new())],
        "$old"
    );
    // New key should be added.
    assert_eq!(
        &base.entries[&("m.room.topic".into(), String::new())],
        "$topic"
    );
}

#[test]
fn state_set_empty_digest_golden_vector() {
    let s = StateSet::new();
    assert_eq!(
        s.digest_base64url(),
        "viqN49z0bJTOhc3I4HrDCPTYqVSQ2VbDjXgP1hDbCBM"
    );
}

/// Pins the unkeyed `LtHash` state-group digest so changing the state-group
/// framing remains a visible, deliberate format break.
#[test]
fn state_set_digest_golden_vector() {
    let mut s = StateSet::new();
    s.set("m.room.name", "", "$old".into());
    assert_eq!(
        s.digest_base64url(),
        "ItsX7KS8GM3yKHTofq3Kp_OQ_lF-zok_yoVyzp_uRig"
    );
}

/// Pins `derive_template_key("blake3-128", ..)`: the first 16 bytes of
/// BLAKE3 of the extracted value, with no domain separation. This is the
/// identity rule behind `matrix_event_node_id`, so the expected value is
/// hard-coded rather than recomputed with the same `sha2` calls.
#[test]
fn derive_template_key_golden_vector() {
    let id = derive_template_key("blake3-128", "$abc123:example.org").unwrap();
    assert_eq!(hex::encode(id), "8dfbdc4a5e4c770e9334d867615a3df9");
    // Only the truncation width is under test here; a different input must
    // produce a different id.
    assert_ne!(
        id,
        derive_template_key("blake3-128", "$other:example.org").unwrap()
    );
}

/// Pins the in-memory DAG `short_id` derivation: the first 8 bytes of
/// BLAKE3(`event_id`) as a little-endian `u64`.
#[test]
fn event_short_id_golden_vector() {
    assert_eq!(
        event_short_id("$abc123:example.org"),
        1_042_385_806_626_192_269
    );
}

#[test]
fn matrix_room_collection_id_matches_internal_derivation() {
    let room_id = "!roomid:example.org";
    assert_eq!(
        hex::encode(matrix_room_collection_id(room_id)),
        "e82141db836a47fedb667d6249aaa94b"
    );
    assert_eq!(
        matrix_room_collection_id(room_id),
        derive_collection_id(MATRIX_ROOM_MEMBER_NAMESPACE, room_id.as_bytes())
    );
}

#[test]
fn multi_dir_shards_iterates_all_databases() {
    let dir1 = unique_temp_dir();
    let dir2 = unique_temp_dir();
    DatabaseLayout::open(dir1.clone()).unwrap();
    DatabaseLayout::open(dir2.clone()).unwrap();

    let cli = Cli {
        dirs: vec![dir1.clone(), dir2.clone()],
        shard_type: None,
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Shards {
            all: false,
            layout: false,
            sort: None,
        },
    };

    cmd_shards(&cli, false, false, None).unwrap();

    std::fs::remove_dir_all(&dir1).unwrap();
    std::fs::remove_dir_all(&dir2).unwrap();
}

#[test]
fn mutating_command_rejects_multi_dir() {
    let dir1 = unique_temp_dir();
    let dir2 = unique_temp_dir();
    let cli = Cli {
        dirs: vec![dir1.clone(), dir2.clone()],
        shard_type: None,
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Put {
            collection: "0x0102030405060708090a0b0c0d0e0f10".to_owned(),
            id: "0x0102030405060708090a0b0c0d0e0f10".to_owned(),
            data: "test".to_owned(),
        },
    };

    let err = super::run(&cli).unwrap_err();
    assert!(
        err.to_string()
            .contains("accepts only a single --dir target"),
        "{err}"
    );
    std::fs::remove_dir_all(&dir1).ok();
    std::fs::remove_dir_all(&dir2).ok();
}

#[test]
fn coalesced_shards_and_collections_and_stats() {
    let dir1 = unique_temp_dir();
    let dir2 = unique_temp_dir();
    let l1 = DatabaseLayout::open(dir1.clone()).unwrap();
    let l2 = DatabaseLayout::open(dir2.clone()).unwrap();

    let s1 = PackfileStorage::open(l1.pool_dir(ShardType::EventDag).unwrap()).unwrap();
    let col = [0x11; 16];
    let n1 = [0x22; 16];
    let d1 = NodeData::new(Bytes::from_static(b"payload 1"));
    s1.put(&col, &n1, &d1).unwrap();
    s1.sync().unwrap();

    let s2 = PackfileStorage::open(l2.pool_dir(ShardType::EventDag).unwrap()).unwrap();
    let n2 = [0x33; 16];
    let d2 = NodeData::new(Bytes::from_static(b"payload 2"));
    s2.put(&col, &n2, &d2).unwrap();
    s2.sync().unwrap();

    let cli_shards = Cli {
        dirs: vec![dir1.clone(), dir2.clone()],
        shard_type: None,
        coalesce: true,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Shards {
            all: false,
            layout: false,
            sort: Some("bytes".to_owned()),
        },
    };
    cmd_shards(&cli_shards, false, false, Some("bytes")).unwrap();

    let cli_cols = Cli {
        dirs: vec![dir1.clone(), dir2.clone()],
        shard_type: None,
        coalesce: true,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Collections {
            per_db: false,
            all: false,
            layout: false,
            canonical: false,
            sort: Some("nodes".to_owned()),
            limit: 10,
        },
    };
    cmd_collections(
        &cli_cols,
        &CollectionOptions {
            flags: CollectionFlags::from_bools([false, false, false, false]),
            sort: Some("nodes"),
            limit: 10,
        },
    )
    .unwrap();
    cmd_collections(
        &cli_cols,
        &CollectionOptions {
            flags: CollectionFlags::from_bools([false, false, false, false]),
            sort: Some("idx-load"),
            limit: 10,
        },
    )
    .unwrap();
    cmd_collections(
        &cli_cols,
        &CollectionOptions {
            flags: CollectionFlags::from_bools([false, false, false, false]),
            sort: Some("load"),
            limit: 10,
        },
    )
    .unwrap();

    let mut cli_cols_coalesce = cli_cols.clone();
    cli_cols_coalesce.coalesce = true;
    cmd_collections(
        &cli_cols_coalesce,
        &CollectionOptions {
            flags: CollectionFlags::from_bools([false, false, false, false]),
            sort: Some("idx-load"),
            limit: 10,
        },
    )
    .unwrap();
    cmd_collections(
        &cli_cols_coalesce,
        &CollectionOptions {
            flags: CollectionFlags::from_bools([false, false, false, false]),
            sort: Some("load"),
            limit: 10,
        },
    )
    .unwrap();

    let cli_stats = Cli {
        dirs: vec![dir1.clone(), dir2.clone()],
        shard_type: None,
        coalesce: true,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Stats { json: true },
    };
    cmd_stats(&cli_stats, true).unwrap();
    cmd_stats(&cli_stats, false).unwrap();

    std::fs::remove_dir_all(&dir1).ok();
    std::fs::remove_dir_all(&dir2).ok();
}

#[test]
fn coalesced_get_resolves_conflict_by_origin_server_ts() {
    let dir1 = unique_temp_dir();
    let dir2 = unique_temp_dir();
    let l1 = DatabaseLayout::open(dir1.clone()).unwrap();
    let l2 = DatabaseLayout::open(dir2.clone()).unwrap();

    let col = [0xAA; 16];
    let node = [0xBB; 16];
    let hex_id = format_id(&node);

    // dir1 has older ts
    let s1 = PackfileStorage::open(l1.pool_dir(ShardType::EventDag).unwrap()).unwrap();
    let old_json = br#"{"origin_server_ts": 1000, "body": "old"}"#;
    s1.put(&col, &node, &NodeData::new(Bytes::from_static(old_json)))
        .unwrap();
    s1.sync().unwrap();

    // dir2 has newer ts
    let s2 = PackfileStorage::open(l2.pool_dir(ShardType::EventDag).unwrap()).unwrap();
    let new_json = br#"{"origin_server_ts": 2000, "body": "new"}"#;
    s2.put(&col, &node, &NodeData::new(Bytes::from_static(new_json)))
        .unwrap();
    s2.sync().unwrap();

    let cli = Cli {
        dirs: vec![dir1.clone(), dir2.clone()],
        shard_type: None,
        coalesce: true,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Get {
            collection: None,
            id: hex_id.clone(),
            raw: true,
            verbose: true,
            header: false,
            decode: None,
        },
    };

    // Coalesced get should succeed and pick the newer candidate
    cmd_get(&cli, None, &hex_id, true, true, false, None).unwrap();

    std::fs::remove_dir_all(&dir1).ok();
    std::fs::remove_dir_all(&dir2).ok();
}

#[test]
fn coalesce_safeguards_reject_unsupported_commands() {
    let dir = unique_temp_dir();
    let _ = DatabaseLayout::open(dir.clone()).unwrap();

    let unsupported = vec![
        Commands::Put {
            collection: "0x00000000000000000000000000000000".to_owned(),
            id: "$event:example.org".to_owned(),
            data: "{}".to_owned(),
        },
        Commands::Delete {
            collections: vec!["0x00000000000000000000000000000000".to_owned()],
            yes: true,
        },
        Commands::Import {
            paths: vec![],
            collection: None,
            template: None,
            no_state_hamts: false,
        },
        Commands::Export {
            collection: "0x00000000000000000000000000000000".to_owned(),
            format: "jsonl".to_owned(),
            metadata: false,
        },
        Commands::Repack {
            collection: None,
            packs: vec![],
            all: true,
            root: vec![],
            topo: false,
            out: None,
            yes: true,
        },
    ];

    for cmd in unsupported {
        let cli = Cli {
            dirs: vec![dir.clone()],
            shard_type: None,
            coalesce: true,
            read_plan: mtxdb::ReadPlanPolicy::disabled(),
            command: cmd,
        };
        assert!(
            run(&cli).is_err(),
            "command should have been rejected with --coalesce"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn coalesced_repack_materializes_canonical_database() {
    let dir1 = unique_temp_dir();
    let dir2 = unique_temp_dir();
    let out_dir = unique_temp_dir();

    let l1 = DatabaseLayout::open(dir1.clone()).unwrap();
    let l2 = DatabaseLayout::open(dir2.clone()).unwrap();

    let col = [0x11; 16];
    let node1 = [0x22; 16];
    let node2 = [0x33; 16];

    let s1 = PackfileStorage::open(l1.pool_dir(ShardType::EventDag).unwrap()).unwrap();
    s1.put(
        &col,
        &node1,
        &NodeData::new(Bytes::from_static(
            br#"{"origin_server_ts": 100, "content": "msg1"}"#,
        )),
    )
    .unwrap();
    s1.sync().unwrap();

    let s2 = PackfileStorage::open(l2.pool_dir(ShardType::EventDag).unwrap()).unwrap();
    s2.put(
        &col,
        &node2,
        &NodeData::new(Bytes::from_static(
            br#"{"origin_server_ts": 200, "content": "msg2"}"#,
        )),
    )
    .unwrap();
    s2.sync().unwrap();

    let cli = Cli {
        dirs: vec![dir1.clone(), dir2.clone()],
        shard_type: Some(ShardType::EventDag),
        coalesce: true,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Repack {
            collection: None,
            packs: vec![],
            all: true,
            root: vec![],
            topo: false,
            out: Some(out_dir.clone()),
            yes: true,
        },
    };

    // Run coalescing repack
    cmd_repack_coalesced(&cli, &out_dir, None, &[], true, &[], false, true).unwrap();

    // Target directory must be a valid canonical database
    let out_layout = DatabaseLayout::open_read_only(out_dir.clone()).unwrap();
    let out_pool = out_layout.pool_dir(ShardType::EventDag).unwrap();
    let out_store = PackfileStorage::open_read_only(out_pool).unwrap();

    // Both nodes must be present in the newly materialized database
    let r1 = out_store.get(&col, &node1).unwrap();
    assert!(r1.is_some());
    let r2 = out_store.get(&col, &node2).unwrap();
    assert!(r2.is_some());

    // Target store must have exactly 1 pack (a freshly materialized one)
    let summaries = out_store.shard_summaries();
    assert_eq!(summaries.len(), 1);

    std::fs::remove_dir_all(&dir1).ok();
    std::fs::remove_dir_all(&dir2).ok();
    std::fs::remove_dir_all(&out_dir).ok();
}

#[test]
fn coalesced_repack_accepts_matrix_room_selectors() {
    let dir = unique_temp_dir();
    let out_dir = unique_temp_dir();
    let layout = DatabaseLayout::open(dir.clone()).unwrap();
    let room = "!selector:example.org";
    let collection_id = matrix_room_collection_id(room);
    let node_id = [0x44; 16];

    let store = PackfileStorage::open(layout.pool_dir(ShardType::EventDag).unwrap()).unwrap();
    store
        .put(
            &collection_id,
            &node_id,
            &NodeData::new(Bytes::from_static(br#"{"origin_server_ts":1}"#)),
        )
        .unwrap();
    store.sync().unwrap();
    drop(store);

    let cli = Cli {
        dirs: vec![dir.clone()],
        shard_type: Some(ShardType::EventDag),
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Repack {
            collection: Some(room.to_owned()),
            packs: vec![],
            all: false,
            root: vec![],
            topo: false,
            out: Some(out_dir.clone()),
            yes: true,
        },
    };
    cmd_repack_coalesced(&cli, &out_dir, Some(room), &[], false, &[], false, true).unwrap();

    let out_layout = DatabaseLayout::open_read_only(out_dir.clone()).unwrap();
    let out_pool = out_layout.pool_dir(ShardType::EventDag).unwrap();
    let out_store = PackfileStorage::open_read_only(out_pool).unwrap();
    assert!(out_store.get(&collection_id, &node_id).unwrap().is_some());

    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&out_dir).ok();
}

#[test]
fn coalesced_info_and_scan_filter_to_matching_databases() {
    let dir1 = unique_temp_dir();
    let dir2 = unique_temp_dir();

    let l1 = DatabaseLayout::open(dir1.clone()).unwrap();
    let l2 = DatabaseLayout::open(dir2.clone()).unwrap();

    let col1 = [0x11; 16];
    let col2 = [0x22; 16];
    let node1 = [0x33; 16];
    let node2 = [0x44; 16];

    let s1 = PackfileStorage::open(l1.pool_dir(ShardType::EventDag).unwrap()).unwrap();
    s1.put(
        &col1,
        &node1,
        &NodeData::new(Bytes::from_static(b"{\"content\": \"msg1\"}")),
    )
    .unwrap();
    s1.sync().unwrap();

    let s2 = PackfileStorage::open(l2.pool_dir(ShardType::EventDag).unwrap()).unwrap();
    s2.put(
        &col2,
        &node2,
        &NodeData::new(Bytes::from_static(b"{\"content\": \"msg2\"}")),
    )
    .unwrap();
    s2.sync().unwrap();

    let cli_info = Cli {
        dirs: vec![dir1.clone(), dir2.clone()],
        shard_type: Some(ShardType::EventDag),
        coalesce: true,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Info {
            selector: Some(format_id(&col1)),
            pack: None,
            collection: None,
            stats: false,
        },
    };
    cmd_info(&cli_info, &format_id(&col1), None, false).unwrap();

    let cli_scan = Cli {
        dirs: vec![dir1.clone(), dir2.clone()],
        shard_type: Some(ShardType::EventDag),
        coalesce: true,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Scan {
            selector: format_id(&col1),
            verbose: false,
            header: false,
            decode: None,
            limit: 10,
            id: None,
            collection: None,
            raw: false,
            sort: None,
            reverse: false,
            live: false,
        },
    };
    cmd_scan(
        &cli_scan,
        &format_id(&col1),
        false,
        false,
        None,
        10,
        None,
        None,
        false,
        None,
        false,
        false,
    )
    .unwrap();

    let missing = format_id(&[0x99; 16]);
    assert!(cmd_scan(
        &cli_scan, &missing, false, false, None, 10, None, None, false, None, false, false
    )
    .is_err());

    std::fs::remove_dir_all(&dir1).ok();
    std::fs::remove_dir_all(&dir2).ok();
}

/// Importing into a root that was only initialized must work: `init` creates just
/// `db.meta`, so the pools do not exist yet and the first write creates them.
#[test]
fn import_into_a_freshly_initialized_root_creates_pools_on_first_write() {
    let dir = unique_temp_dir();
    let layout = DatabaseLayout::open(dir.clone()).unwrap();
    for shard_type in ShardType::ALL {
        assert!(
            !layout.pool_path(shard_type).exists(),
            "a fresh root has no pool directories"
        );
    }
    let input = dir.join("export.json");
    std::fs::write(
        &input,
        r#"{
            "pdus": [
                {"event_id":"$c","room_id":"!room","sender":"@server",
                 "type":"m.room.create","state_key":"",
                 "content":{"creator":"@server","room_version":"10"},
                 "auth_events":[],"prev_events":[]},
                {"event_id":"$m","room_id":"!room","sender":"@server",
                 "type":"m.room.message","content":{},
                 "auth_events":["$c"],"prev_events":["$c"]}
            ]
        }"#,
    )
    .unwrap();
    let cli = Cli {
        dirs: vec![dir.clone()],
        shard_type: Some(ShardType::EventDag),
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Import {
            paths: vec![input.clone()],
            collection: None,
            template: None,
            no_state_hamts: false,
        },
    };

    cmd_import(&cli, &[input], None, None, false).expect("import into a fresh root");

    assert!(
        layout.pool_path(ShardType::EventDag).is_dir(),
        "the event pool exists once it was written"
    );
    assert!(
        !glob_pack_files(&layout.pool_path(ShardType::EventDag))
            .unwrap()
            .is_empty(),
        "the import wrote a pack"
    );
    // Read-only inspection of the pools that were never written still works.
    let inspect = Cli {
        dirs: vec![dir.clone()],
        shard_type: None,
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Shards {
            all: false,
            layout: false,
            sort: None,
        },
    };
    cmd_shards(&inspect, false, false, None).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A missing database root is an error that says so; a valid root whose pool was
/// never written is just empty, and inspecting it creates nothing.
#[test]
fn a_missing_root_is_an_error_but_an_unwritten_pool_is_empty() {
    let missing = unique_temp_dir();
    let cli = |dir: &Path| Cli {
        dirs: vec![dir.to_path_buf()],
        shard_type: Some(ShardType::State),
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Collections {
            per_db: false,
            all: false,
            layout: false,
            canonical: false,
            sort: None,
            limit: 0,
        },
    };
    let error = cmd_collections(
        &cli(&missing),
        &CollectionOptions {
            flags: CollectionFlags::from_bools([false, false, false, false]),
            sort: None,
            limit: 0,
        },
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("no mtxdb database"),
        "a missing root must be diagnosed as such: {error}"
    );
    assert!(!missing.exists(), "inspection must not create the root");

    let root = unique_temp_dir();
    let layout = DatabaseLayout::open(root.clone()).unwrap();
    cmd_collections(
        &cli(&root),
        &CollectionOptions {
            flags: CollectionFlags::from_bools([false, false, false, false]),
            sort: None,
            limit: 0,
        },
    )
    .expect("an unwritten pool is empty, not an error");
    assert!(
        !layout.pool_path(ShardType::State).exists(),
        "read-only inspection must not create the pool"
    );
    std::fs::remove_dir_all(&root).unwrap();
}

/// A batch that declares a room version gets real Matrix state resolution: the
/// merged state is decided by power levels, not by `prev_events` order. The
/// retired first-parent merge gives the opposite answer for the same fork
/// listed in the other order, which is the whole point of the resolver.
#[test]
fn real_resolution_decides_a_fork_by_power_level_not_prev_order() {
    let create = owned_value(
        r#"{"event_id":"$create","room_id":"!r:x","type":"m.room.create","state_key":"","sender":"@admin:x","depth":1,"prev_events":[],"auth_events":[],"content":{"room_version":"10","creator":"@admin:x"}}"#,
    );
    let member = owned_value(
        r#"{"event_id":"$admin-member","room_id":"!r:x","type":"m.room.member","state_key":"@admin:x","sender":"@admin:x","depth":2,"prev_events":["$create"],"auth_events":["$create"],"content":{"membership":"join"}}"#,
    );
    let power_levels = owned_value(
        r#"{"event_id":"$pl","room_id":"!r:x","type":"m.room.power_levels","state_key":"","sender":"@admin:x","depth":3,"prev_events":["$admin-member"],"auth_events":["$create","$admin-member"],"content":{"users_default":0,"state_default":50,"events_default":0,"users":{"@admin:x":100}}}"#,
    );
    let admin_topic = owned_value(
        r#"{"event_id":"$topic-admin","room_id":"!r:x","type":"m.room.topic","state_key":"","sender":"@admin:x","depth":4,"prev_events":["$pl"],"auth_events":["$create","$admin-member","$pl"],"content":{"topic":"kept"}}"#,
    );
    let nobody_topic = owned_value(
        r#"{"event_id":"$topic-nobody","room_id":"!r:x","type":"m.room.topic","state_key":"","sender":"@nobody:x","depth":4,"prev_events":["$pl"],"auth_events":["$create","$pl"],"content":{"topic":"dropped"}}"#,
    );
    let merge = |first: &str, second: &str, id: &str| {
        owned_value(&format!(
            r#"{{"event_id":"{id}","room_id":"!r:x","type":"m.room.message","sender":"@admin:x","depth":5,"prev_events":["{first}","{second}"],"auth_events":["$create","$admin-member","$pl"],"content":{{}}}}"#
        ))
    };
    let batch = |first: &str, second: &str, id: &str| {
        vec![
            create.clone(),
            member.clone(),
            power_levels.clone(),
            admin_topic.clone(),
            nobody_topic.clone(),
            merge(first, second, id),
        ]
    };

    let forward = compute_state_groups_partial(
        &batch("$topic-admin", "$topic-nobody", "$merge-forward"),
        &[],
        "!r:x",
    );
    let reverse = compute_state_groups_partial(
        &batch("$topic-nobody", "$topic-admin", "$merge-reverse"),
        &[],
        "!r:x",
    );
    assert!(
        forward.incomplete.is_empty(),
        "batch carries the whole auth closure: {:?}",
        forward.incomplete
    );
    assert!(
        reverse.incomplete.is_empty(),
        "batch carries the whole auth closure: {:?}",
        reverse.incomplete
    );

    let forward_id = forward.groups["$merge-forward"];
    let reverse_id = reverse.groups["$merge-reverse"];
    assert_eq!(
        forward_id, reverse_id,
        "prev_events order must not change the resolved state"
    );

    let resolved = &forward.instances[&forward_id];
    let topic = resolved
        .set
        .entries
        .get(&("m.room.topic".to_owned(), String::new()))
        .map(String::as_str);
    assert_eq!(
        topic,
        Some("$topic-admin"),
        "the sender with power over state wins the conflict"
    );
}

/// A merge whose inputs reference an event outside the batch cannot run the
/// resolver: mtxdb's binary is `panic = "abort"`, so it must fall back rather
/// than let rezzy look the id up. The fallback still yields a group, but the
/// computation is marked incomplete so the import does not cache it as a real
/// resolution.
#[test]
fn merges_with_a_gap_in_the_auth_closure_fall_back_and_refuse_to_cache() {
    let create = owned_value(
        r#"{"event_id":"$create","room_id":"!r:x","type":"m.room.create","state_key":"","sender":"@admin:x","depth":1,"prev_events":[],"auth_events":[],"content":{"room_version":"10"}}"#,
    );
    let power_levels = owned_value(
        r#"{"event_id":"$pl","room_id":"!r:x","type":"m.room.power_levels","state_key":"","sender":"@admin:x","depth":2,"prev_events":["$create"],"auth_events":["$create"],"content":{"users":{"@admin:x":100}}}"#,
    );
    let complete_topic = owned_value(
        r#"{"event_id":"$topic-a","room_id":"!r:x","type":"m.room.topic","state_key":"","sender":"@admin:x","depth":3,"prev_events":["$pl"],"auth_events":["$create","$pl"],"content":{"topic":"first"}}"#,
    );
    let gapped_topic = owned_value(
        r#"{"event_id":"$topic-b","room_id":"!r:x","type":"m.room.topic","state_key":"","sender":"@admin:x","depth":3,"prev_events":["$pl"],"auth_events":["$create","$pl","$absent-auth"],"content":{"topic":"second"}}"#,
    );
    let merge = owned_value(
        r#"{"event_id":"$merge","room_id":"!r:x","type":"m.room.message","sender":"@admin:x","depth":4,"prev_events":["$topic-a","$topic-b"],"auth_events":["$create","$pl"],"content":{}}"#,
    );
    let events = vec![create, power_levels, complete_topic, gapped_topic, merge];

    let result = compute_state_groups_partial(&events, &[], "!r:x");
    assert_eq!(
        result.incomplete,
        vec!["$absent-auth".to_owned()],
        "the resolver reports exactly what the batch is short of"
    );
    assert!(
        result.groups.contains_key("$merge"),
        "the merge still gets a group from the fallback"
    );
    let resolved = &result.instances[&result.groups["$merge"]];
    assert_eq!(
        resolved
            .set
            .entries
            .get(&("m.room.topic".to_owned(), String::new()))
            .map(String::as_str),
        Some("$topic-a"),
        "the fallback keeps the first parent, as the retired merge did"
    );
}

fn state_cli(dir: &std::path::Path, command: Commands) -> Cli {
    Cli {
        dirs: vec![dir.to_path_buf()],
        shard_type: Some(ShardType::State),
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command,
    }
}

#[test]
fn remove_hides_records_and_tombstones_reports_them() {
    let dir = unique_temp_dir();
    let layout = DatabaseLayout::open(dir.clone()).unwrap();
    let state_dir = layout.pool_dir(ShardType::State).unwrap();
    let collection = [0x31u8; 16];
    let kept = [0x01u8; 16];
    let gone = [0x02u8; 16];
    {
        let store = PackfileStorage::open(state_dir.clone()).unwrap();
        for id in [kept, gone] {
            store
                .put(
                    &collection,
                    &id,
                    &NodeData::new(Bytes::from_static(b"value")),
                )
                .unwrap();
        }
        store.sync().unwrap();
    }

    let collection_hex = format_id(&collection);
    let cli = state_cli(
        &dir,
        Commands::Remove {
            collection: collection_hex.clone(),
            ids: vec![format_id(&gone), format_id(&[0x7fu8; 16])],
            yes: true,
        },
    );
    // The second id was never live: it is skipped, not an error.
    cmd_remove(
        &cli,
        &collection_hex,
        &[format_id(&gone), format_id(&[0x7fu8; 16])],
        true,
    )
    .unwrap();

    let store = PackfileStorage::open_read_only(state_dir.clone()).unwrap();
    assert!(store.get(&collection, &gone).unwrap().is_none());
    assert!(store.get(&collection, &kept).unwrap().is_some());
    let stats = store.removal_stats(&collection).unwrap();
    assert_eq!(stats.live_keys, 1);
    assert_eq!(stats.removed_keys, 1);
    assert_eq!(store.removed_keys(&collection).unwrap(), vec![gone]);
    drop(store);

    let report = state_cli(
        &dir,
        Commands::Tombstones {
            collection: collection_hex.clone(),
            keys: true,
            detail: true,
        },
    );
    // Counts only (index), counts plus frame totals, and the key listing.
    cmd_tombstones(&report, &collection_hex, false, false).unwrap();
    cmd_tombstones(&report, &collection_hex, false, true).unwrap();
    cmd_tombstones(&report, &collection_hex, true, true).unwrap();
    // An unknown collection is an error, not an empty report.
    assert!(cmd_tombstones(&report, &format_id(&[0x77u8; 16]), false, false).is_err());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn remove_rejects_a_malformed_id_before_touching_the_store() {
    let dir = unique_temp_dir();
    let layout = DatabaseLayout::open(dir.clone()).unwrap();
    let state_dir = layout.pool_dir(ShardType::State).unwrap();
    let collection = [0x32u8; 16];
    let id = [0x01u8; 16];
    {
        let store = PackfileStorage::open(state_dir.clone()).unwrap();
        store
            .put(
                &collection,
                &id,
                &NodeData::new(Bytes::from_static(b"value")),
            )
            .unwrap();
        store.sync().unwrap();
    }
    let collection_hex = format_id(&collection);
    let cli = state_cli(
        &dir,
        Commands::Remove {
            collection: collection_hex.clone(),
            ids: vec![format_id(&id), "not-hex".to_owned()],
            yes: true,
        },
    );
    assert!(cmd_remove(
        &cli,
        &collection_hex,
        &[format_id(&id), "not-hex".to_owned()],
        true
    )
    .is_err());
    let store = PackfileStorage::open_read_only(state_dir).unwrap();
    assert!(
        store.get(&collection, &id).unwrap().is_some(),
        "a bad id must fail the whole command before any record is removed"
    );
    drop(store);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The exporter's own output, copied from Continuwuity's
/// `tests/fixtures/mtxdb-export/` (its tests assert the exporter still
/// produces exactly these lines). Importing these files, not a lookalike, is
/// what shows both sides agree.
const ROOM_EXPORT_FIXTURE: &str = include_str!("../tests/fixtures/mtxdb-export/room-v1.jsonl");
const KEYS_EXPORT_FIXTURE: &str =
    include_str!("../tests/fixtures/mtxdb-export/server-keys-v1.jsonl");

const FIXTURE_ROOM: &str = "!export:test.conduwuit.local";

/// Replace the fixture's placeholder content hashes with the hashes its PDUs
/// actually have. The exporter fixture uses `"sha256":"dummy"` because its
/// byte-for-byte test does not run the hash check; the importer does, and a
/// mismatch now stores the redacted form. The layout test wants the unchanged
/// body, so make the hashes valid. The mismatch path has its own test.
fn with_valid_content_hashes(content: &str) -> String {
    let room_version = content
        .lines()
        .filter_map(|line| OwnedValue::parse_bytes(line.as_bytes()).ok())
        .find_map(|value| {
            let pdu = value.get("pdu")?;
            (pdu.get("type")?.as_str()? == "m.room.create")
                .then(|| {
                    pdu.get("content")?
                        .get("room_version")?
                        .as_str()
                        .map(str::to_owned)
                })
                .flatten()
        })
        .expect("layout fixture create event has a room version");

    content
        .lines()
        .map(|line| {
            if !line.contains("\"t\":\"event\"") {
                return line.to_owned();
            }
            let value = OwnedValue::parse_bytes(line.as_bytes()).expect("valid fixture line");
            let pdu = value.get("pdu").expect("fixture event has a pdu");
            let hash = rezzy::compute_content_hash(pdu, &room_version)
                .expect("fixture content hash is computable");
            line.replacen("\"sha256\":\"dummy\"", &format!("\"sha256\":\"{hash}\""), 1)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The exporter fixture with the auth chain a real export carries: a member
/// event for the creator, cited alongside the create event by every other
/// event. The exporter's own fixture has empty auth lists, which is right for
/// its byte-for-byte test but describes a room the auth rules refuse.
///
/// Every event passes the first auth check, so `$rejected` is a deliberate
/// disagreement: the exporter rejected it, but the first check does not.
fn room_export_with_auth() -> String {
    const CREATE: &str = "$create:test.conduwuit.local";
    const JOIN: &str = "$join:test.conduwuit.local";

    let join = format!(
        r#"{{"auth_events":["{CREATE}"],"event_id":"{JOIN}","outlier":false,"prev_events":["{CREATE}"],"pdu":{{"auth_events":["{CREATE}"],"content":{{"membership":"join"}},"depth":1,"event_id":"{JOIN}","origin_server_ts":123456789,"prev_events":["{CREATE}"],"room_id":"!export:test.conduwuit.local","sender":"@alice:test.conduwuit.local","state_key":"@alice:test.conduwuit.local","type":"m.room.member"}},"rejected":false,"soft_failed":false,"t":"event"}}"#,
    );

    let mut lines = Vec::new();
    for line in ROOM_EXPORT_FIXTURE.lines() {
        if line.starts_with("{\"events\":") {
            lines.push(join.clone());
            lines.push(line.replace("\"events\":5", "\"events\":6"));
            continue;
        }
        if !line.contains("\"t\":\"event\"") || line.contains(CREATE) {
            lines.push(line.to_owned());
            continue;
        }
        let auth = format!("[\"{JOIN}\",\"{CREATE}\"]");
        let line = line.replace("\"auth_events\":[]", &format!("\"auth_events\":{auth}"));
        let line = line.replace(
            "\"prev_events\":[]",
            &format!("\"prev_events\":[\"{JOIN}\"]"),
        );
        lines.push(line);
    }
    with_valid_content_hashes(&lines.join("\n"))
}

/// Import `inputs` the way `mtxdb import` does, minus the final phase.
fn import_inputs(db: &Database, root: &std::path::Path, inputs: &[(&str, &str)]) -> ImportRun {
    let paths: Vec<PathBuf> = inputs
        .iter()
        .map(|(name, content)| {
            let path = root.join(name);
            std::fs::write(&path, content).unwrap();
            path
        })
        .collect();
    let store = db.event_dag();
    let template = default_matrix_import_template();
    import_data(
        db,
        store,
        store,
        &paths,
        None,
        &template,
        &mut HashSet::new(),
        true,
    )
}

fn stored(
    db: &Database,
    pool: mtxdb::ShardType,
    collection: [u8; 16],
    node: [u8; 16],
) -> Option<Vec<u8>> {
    use mtxdb::storage::StorageEngine as _;
    db.pool(pool)
        .get(&collection, &node)
        .unwrap()
        .map(|data| data.bytes.to_vec())
}

#[test]
fn imported_room_export_is_readable_in_the_servers_layout() {
    use crate::interchange::parse_room;
    use mtxdb::interop::*;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let content = room_export_with_auth();
    let run = import_inputs(&db, &root, &[("room.jsonl", &content)]);
    assert_eq!(
        (run.failures, run.all_exports, run.exported_events),
        (0, true, 6)
    );
    finish_import(&db, &run).unwrap();

    // The fixture with a creator membership: the export marks `$soft`
    // soft-failed and `$rejected` rejected. Rejection is derived and not
    // migrated, but the soft-fail is kept as a source fact. `$redacted` was
    // redacted by `$redaction`, which is a fact about the stored event, and
    // `$ok` carries no verdict.
    let parsed = parse_room(content.as_bytes()).unwrap();
    assert_eq!(parsed.room_id, FIXTURE_ROOM);
    assert_eq!(parsed.events.len(), 6);
    let room_bytes = FIXTURE_ROOM.as_bytes();
    for event in &parsed.events {
        let id = event.event_id.as_bytes();
        let record = event_record_id(id);

        // The body is the export's PDU, byte for byte. The fixture's hashes are
        // valid (or absent), so no event here takes the redaction path.
        let body = stored(
            &db,
            EVENT_POOL,
            room_event_collection_id(room_bytes),
            record,
        )
        .expect("event body");
        assert_eq!(
            body,
            event.pdu_bytes.as_ref(),
            "{} must keep the export's bytes",
            event.event_id
        );

        // The locator maps the event back to its room.
        let locator = stored(
            &db,
            EVENT_POOL,
            event_locator_collection_id(&record),
            record,
        )
        .expect("locator");
        assert_eq!(decode_event_locator(&locator).unwrap(), (id, room_bytes));
    }

    let verdict = |name: &str| {
        let id = format!("${name}:test.conduwuit.local");
        let record = event_record_id(id.as_bytes());
        stored(
            &db,
            EVENT_POOL,
            event_verdict_collection_id(&record),
            record,
        )
        .map(|bytes| {
            let (event, verdict) = decode_verdict(&bytes).unwrap();
            assert_eq!(event, id.as_bytes());
            verdict
        })
    };
    assert_eq!(
        verdict("soft").unwrap().soft_fail,
        Some(2),
        "an exported soft-fail is kept with its reason"
    );
    assert!(
        verdict("rejected").is_none(),
        "an exported rejection is not migrated; it passes the derived check"
    );
    assert_eq!(
        verdict("redacted").unwrap(),
        Verdict {
            redacted_by: Some(b"$redaction:test.conduwuit.local".to_vec()),
            ..Verdict::default()
        }
    );
    assert!(
        verdict("ok").is_none(),
        "a clean timeline event needs no verdict record"
    );
}

#[test]
fn an_imported_soft_fail_sets_the_soft_failed_flag() {
    use mtxdb::room_flag::*;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let content = room_export_with_auth();
    let run = import_inputs(&db, &root, &[("room.jsonl", &content)]);
    assert_eq!((run.failures, run.all_exports), (0, true));

    let flags = RoomFlagIndex::new(FIXTURE_ROOM.as_bytes());
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::SoftFailed)).unwrap(),
        1,
        "a kept soft-fail is indexed as soft-failed"
    );
}

#[test]
fn an_exported_outlier_flag_is_not_migrated() {
    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    // The exporter marks `$ok` an outlier; the stored verdict omits the
    // annotation, while the rebuild must recover the derived flag from it.
    let content = room_export_with_auth().replace(
        "\"event_id\":\"$ok:test.conduwuit.local\",\"outlier\":false",
        "\"event_id\":\"$ok:test.conduwuit.local\",\"outlier\":true",
    );
    assert_ne!(
        content,
        room_export_with_auth(),
        "the fixture line this test edits moved"
    );
    import_inputs(&db, &root, &[("room.jsonl", &content)]);

    let outlier = |name: &str| {
        let id = format!("${name}:test.conduwuit.local");
        let record = mtxdb::interop::event_record_id(id.as_bytes());
        stored(
            &db,
            mtxdb::interop::EVENT_POOL,
            mtxdb::interop::event_verdict_collection_id(&record),
            record,
        )
        .map(|bytes| mtxdb::interop::decode_verdict(&bytes).unwrap().1.outlier)
    };
    assert_eq!(
        outlier("ok"),
        Some(false),
        "an exported outlier is not migrated"
    );
    assert_eq!(
        outlier("soft"),
        Some(false),
        "a soft-failed event has a verdict but is not a derived outlier"
    );
    assert_eq!(outlier("rejected"), None);
    assert_eq!(
        outlier("redacted"),
        Some(false),
        "a redacted timeline event is not an outlier"
    );
}

#[test]
fn imported_keys_export_is_readable_in_the_servers_layout() {
    use mtxdb::server_keys::{read_notary_payload, read_server_keys, KeyTier};

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let run = import_inputs(&db, &root, &[("keys.jsonl", KEYS_EXPORT_FIXTURE)]);
    assert_eq!(
        (run.failures, run.all_exports, run.exported_events),
        (0, true, 0)
    );

    let mut expected_keys = Vec::new();
    let mut expected_notary = None;
    for line in KEYS_EXPORT_FIXTURE.lines() {
        let record = owned_value(line);
        let decode = |field: &str| {
            base64::engine::general_purpose::STANDARD
                .decode(json_str(&record, field).unwrap())
                .unwrap()
        };
        match json_str(&record, "kind").unwrap().as_str() {
            "notary" => expected_notary = Some(decode("payload")),
            kind => expected_keys.push((
                json_str(&record, "key_id").unwrap().into_bytes(),
                record.get("key").map_or_else(Vec::new, |_| decode("key")),
                json_str(&record, "source").unwrap().into_bytes(),
                kind == "permanent",
                record.get("expired_ts").and_then(OwnedValue::as_u64),
                record
                    .get("promote_by_ms")
                    .and_then(OwnedValue::as_u64)
                    .unwrap_or(0),
                record.get("digest").is_some(),
                false,
            )),
        }
    }

    let stored = read_server_keys(db.server_info(), b"keys.example.org").unwrap();
    let actual: Vec<_> = stored
        .iter()
        .map(|key| {
            (
                key.key_id.clone(),
                key.key.clone(),
                key.source.clone(),
                key.tier == KeyTier::Permanent,
                key.expired_ts,
                key.promote_by_ms,
                key.is_evicted(),
                key.corroborated,
            )
        })
        .collect();
    expected_keys.sort();
    assert_eq!(actual, expected_keys, "each key comes back field for field");
    assert_eq!(
        read_notary_payload(db.server_info(), b"keys.example.org").unwrap(),
        expected_notary,
        "the notary payload must come back byte for byte"
    );
    assert_eq!(
        read_server_keys(db.server_info(), b"other.example.org").unwrap(),
        Vec::<mtxdb::server_keys::ServerKey>::new()
    );
}

fn marker(db: &Database) -> Option<mtxdb::interop::ImportMarker> {
    export_import::read_marker(db).unwrap()
}

#[test]
fn a_keys_export_is_detected_after_leading_blank_lines() {
    let root = unique_temp_dir();
    std::fs::create_dir_all(&root).unwrap();
    let room_path = root.join("room.jsonl");
    let keys_path = root.join("keys.jsonl");
    std::fs::write(&room_path, ROOM_EXPORT_FIXTURE).unwrap();
    std::fs::write(&keys_path, format!("\n\n{KEYS_EXPORT_FIXTURE}")).unwrap();

    assert!(is_server_keys_export(&keys_path));
    assert!(!is_server_keys_export(&room_path));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn an_import_stopped_before_the_marker_is_visibly_incomplete() {
    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let run = import_inputs(&db, &root, &[("room.jsonl", ROOM_EXPORT_FIXTURE)]);

    // Every event is committed, and the run stops before its last step.
    assert_eq!(run.exported_events, 5);
    assert_eq!(marker(&db), None, "no marker until the final phase commits");

    finish_import(&db, &run).unwrap();
    assert_eq!(
        marker(&db),
        Some(mtxdb::interop::ImportMarker {
            data_format: mtxdb::interop::DATA_FORMAT,
            events: 5,
            closures: 5,
        })
    );
}

#[test]
fn a_new_run_withdraws_the_previous_marker_until_it_finishes() {
    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let first = import_inputs(&db, &root, &[("room.jsonl", ROOM_EXPORT_FIXTURE)]);
    finish_import(&db, &first).unwrap();
    assert!(marker(&db).is_some());

    // A second run that is interrupted after its data must not be vouched for
    // by the first run's marker.
    let second = import_inputs(&db, &root, &[("room.jsonl", ROOM_EXPORT_FIXTURE)]);
    assert_eq!(marker(&db), None);
    finish_import(&db, &second).unwrap();
    assert_eq!(
        marker(&db).unwrap().events,
        5,
        "the count is this run's, not a running total"
    );
}

#[test]
fn a_failed_input_means_no_marker() {
    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let run = import_inputs(
        &db,
        &root,
        &[
            ("room.jsonl", ROOM_EXPORT_FIXTURE),
            (
                "broken.jsonl",
                "{\"t\":\"header\",\"format\":\"mtxdb-export/v1\"",
            ),
        ],
    );
    assert_eq!(run.failures, 1);
    assert!(!run.all_exports);
    finish_import(&db, &run).unwrap();
    assert_eq!(
        marker(&db),
        None,
        "a run with a failed input is never marked complete"
    );
}

#[test]
fn server_keys_are_imported_before_rooms_whatever_the_argument_order() {
    let root = unique_temp_dir();
    std::fs::create_dir_all(&root).unwrap();
    let write = |name: &str, content: &str| {
        let path = root.join(name);
        std::fs::write(&path, content).unwrap();
        path
    };
    let room_path = write("a-room.jsonl", ROOM_EXPORT_FIXTURE);
    let keys_path = write("z-keys.jsonl", KEYS_EXPORT_FIXTURE);
    let other_room_path = write("b-room.jsonl", ROOM_EXPORT_FIXTURE);
    let inputs = [
        room_path.clone(),
        keys_path.clone(),
        other_room_path.clone(),
    ];
    assert_eq!(
        keys_first(&inputs),
        [&keys_path, &room_path, &other_room_path]
    );
}

#[test]
fn a_malformed_keys_only_run_clears_an_earlier_runs_marker() {
    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let good = import_inputs(&db, &root, &[("room.jsonl", &room_export_with_auth())]);
    finish_import(&db, &good).unwrap();
    assert!(marker(&db).is_some(), "the first run is marked complete");

    let bad = import_inputs(
        &db,
        &root,
        &[(
            "keys.jsonl",
            "{\"kind\":\"bogus\",\"origin\":\"keys.example.org\",\"t\":\"server_keys\"}",
        )],
    );
    finish_import(&db, &bad).unwrap();
    assert_eq!(
        marker(&db),
        None,
        "a failed run must not leave an earlier run's marker in place"
    );
}

#[test]
fn a_failed_keys_import_is_flagged_and_leaves_no_marker() {
    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let run = import_inputs(
        &db,
        &root,
        &[
            ("room.jsonl", ROOM_EXPORT_FIXTURE),
            (
                "keys.jsonl",
                "{\"kind\":\"bogus\",\"origin\":\"keys.example.org\",\"t\":\"server_keys\"}",
            ),
        ],
    );
    assert_eq!(run.failures, 1);
    assert!(
        !run.signature_keys_available,
        "a failed key import into an empty store leaves signature keys unavailable"
    );
    finish_import(&db, &run).unwrap();
    assert_eq!(marker(&db), None);
}

#[test]
fn a_failed_key_import_keeps_previously_loaded_signing_keys_available() {
    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let first = import_inputs(&db, &root, &[("keys.jsonl", KEYS_EXPORT_FIXTURE)]);
    assert_eq!(first.failures, 0);

    let run = import_inputs(
        &db,
        &root,
        &[
            ("room.jsonl", ROOM_EXPORT_FIXTURE),
            (
                "keys.jsonl",
                "{\"kind\":\"bogus\",\"origin\":\"keys.example.org\",\"t\":\"server_keys\"}",
            ),
        ],
    );
    assert_eq!(run.failures, 1);
    assert!(
        run.signature_keys_available,
        "a failed new key input must not hide usable keys already stored"
    );
    finish_import(&db, &run).unwrap();
    assert_eq!(marker(&db), None, "the failed input keeps the run unmarked");
}

#[test]
fn a_malformed_room_import_is_flagged_and_leaves_no_signing_keys() {
    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let run = import_inputs(
        &db,
        &root,
        &[(
            "broken.jsonl",
            "{\"t\":\"header\",\"format\":\"mtxdb-export/v1\"",
        )],
    );
    assert_eq!(run.failures, 1);
    assert!(!run.signature_keys_available);
}

#[test]
fn a_room_import_in_capped_batches_writes_every_event() {
    use crate::interchange::parse_room;
    use mtxdb::interop::*;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let export = parse_room(ROOM_EXPORT_FIXTURE.as_bytes()).unwrap();
    let largest = export
        .events
        .iter()
        .map(|e| e.pdu_bytes.len())
        .max()
        .unwrap();

    // Room for about two events per batch, so the room takes several commits.
    let cap = largest.saturating_mul(2).saturating_add(2048);
    let (written, _) = export_import::write_room_with_findings(&db, &export, cap).unwrap();
    assert_eq!((written.events, written.closures), (5, 5));
    for event in &export.events {
        let record = event_record_id(event.event_id.as_bytes());
        assert!(
            stored(
                &db,
                EVENT_POOL,
                room_event_collection_id(FIXTURE_ROOM.as_bytes()),
                record
            )
            .is_some(),
            "{} must be committed",
            event.event_id
        );
    }

    // One event larger than a batch is an error, not a silent overrun.
    let error = export_import::write_room_with_findings(&db, &export, 64)
        .unwrap_err()
        .to_string();
    assert!(error.contains("batch limit"), "{error}");
}

#[test]
fn a_room_export_without_a_create_event_is_refused() {
    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let without_create = ROOM_EXPORT_FIXTURE
        .lines()
        .filter(|line| !line.contains("m.room.create"))
        .collect::<Vec<_>>()
        .join("\n");
    let run = import_inputs(&db, &root, &[("room.jsonl", &without_create)]);
    assert_eq!(run.failures, 1, "{run:?}");
    assert_eq!(run.exported_events, 0);
}

#[test]
fn a_room_import_populates_timeline_index() {
    use crate::interchange::parse_room;
    use mtxdb::timeline::*;
    use mtxdb::ShardType;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let content = room_export_with_auth();
    let export = parse_room(content.as_bytes()).unwrap();

    let (written, _) = export_import::write_room_with_findings(&db, &export, 64 * 1024).unwrap();
    assert_eq!(written.events, 6);

    let room_scope = room_scope_id(FIXTURE_ROOM.as_bytes());
    let timeline_index = TimelineIndex::open(db.pool(ShardType::EventDag).as_ref(), FIXTURE_ROOM);
    let page = timeline_index.page(&room_scope, None, true, 10).unwrap();
    assert_eq!(page.entries.len(), 6);

    for pair in page.entries.windows(2) {
        assert!(pair[0] < pair[1]);
    }
}

#[test]
fn a_room_import_populates_flag_index() {
    use crate::interchange::parse_room;
    use mtxdb::room_flag::*;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let content = room_export_with_auth();
    let export = parse_room(content.as_bytes()).unwrap();

    let (written, _) = export_import::write_room_with_findings(&db, &export, 64 * 1024).unwrap();
    assert_eq!(written.events, 6);

    let flag_index = RoomFlagIndex::new(FIXTURE_ROOM.as_bytes());
    assert_eq!(
        flag_index
            .count(&db, &FlagQuery::has(Flag::Present))
            .unwrap(),
        6
    );
    assert_eq!(
        flag_index
            .count(&db, &FlagQuery::has(Flag::Timeline))
            .unwrap(),
        6
    );
    assert_eq!(
        flag_index
            .count(&db, &FlagQuery::has(Flag::Outlier))
            .unwrap(),
        0
    );
    assert_eq!(
        flag_index
            .count(&db, &FlagQuery::has(Flag::Rejected))
            .unwrap(),
        0
    );
    assert_eq!(
        flag_index
            .count(&db, &FlagQuery::has(Flag::SoftFailed))
            .unwrap(),
        1
    );
    assert_eq!(
        flag_index
            .count(&db, &FlagQuery::has(Flag::Redacted))
            .unwrap(),
        1
    );

    let selected = flag_index
        .select(&db, &FlagQuery::has(Flag::Present))
        .unwrap();
    assert_eq!(selected.len(), 6);
}

#[test]
fn flags_cli_command_works() {
    use crate::interchange::parse_room;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let export = parse_room(ROOM_EXPORT_FIXTURE.as_bytes()).unwrap();
    let _ = export_import::write_room_with_findings(&db, &export, 64 * 1024).unwrap();
    drop(db);

    let cli = Cli {
        dirs: vec![root.clone()],
        shard_type: None,
        coalesce: false,
        read_plan: mtxdb::ReadPlanPolicy::disabled(),
        command: Commands::Flags {
            room: FIXTURE_ROOM.to_owned(),
            flag: Some("timeline".to_owned()),
            not: false,
            count: true,
            limit: 10,
            rebuild: false,
        },
    };

    cmd_flags(&cli, FIXTURE_ROOM, Some("timeline"), false, true, 10, false).unwrap();
    cmd_flags(
        &cli,
        FIXTURE_ROOM,
        Some("timeline"),
        false,
        false,
        10,
        false,
    )
    .unwrap();
    cmd_flags(&cli, FIXTURE_ROOM, None, false, false, 10, false).unwrap();
    cmd_flags(&cli, FIXTURE_ROOM, None, false, false, 10, true).unwrap();
}

#[test]
fn forced_rejection_import_sets_flags_and_not_excludes_it() {
    use crate::interchange::parse_room;
    use mtxdb::room_flag::*;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();

    let room_id = "!rejection_test:test.conduwuit.local";
    let jsonl = format!(
        concat!(
            r#"{{"format":"mtxdb-export/v1","room_id":"{room}","server":"test.conduwuit.local","timeline":true,"t":"header"}}"#,
            "\n",
            r#"{{"event_id":"$create:test","depth":1,"stream_ordering":1,"auth_events":[],"prev_events":[],"outlier":false,"pdu":{{"auth_events":[],"content":{{"room_version":"10"}},"depth":1,"event_id":"$create:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":1000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","state_key":"","type":"m.room.create"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$join:test","depth":2,"stream_ordering":2,"auth_events":["$create:test"],"prev_events":["$create:test"],"outlier":false,"pdu":{{"auth_events":["$create:test"],"content":{{"membership":"join"}},"depth":2,"event_id":"$join:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":2000,"prev_events":["$create:test"],"room_id":"{room}","sender":"@alice:test","state_key":"@alice:test","type":"m.room.member"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$bad:test","depth":3,"stream_ordering":3,"auth_events":["$create:test","$join:test","$missing:test"],"prev_events":["$join:test"],"outlier":false,"pdu":{{"auth_events":["$create:test","$join:test","$missing:test"],"content":{{}},"depth":3,"event_id":"$bad:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":3000,"prev_events":["$join:test"],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"events":3,"t":"end"}}"#,
            "\n"
        ),
        room = room_id
    );

    let export = parse_room(jsonl.as_bytes()).unwrap();
    let (written, findings) =
        export_import::write_room_with_findings(&db, &export, 64 * 1024).unwrap();
    assert_eq!(written.events, 3);
    assert_eq!(findings.rejection_counts.values().sum::<usize>(), 1);

    let flags = RoomFlagIndex::new(room_id.as_bytes());
    assert_eq!(flags.count(&db, &FlagQuery::has(Flag::Present)).unwrap(), 3);
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Rejected)).unwrap(),
        1
    );
    assert_eq!(
        flags.count(&db, &FlagQuery::not(Flag::Rejected)).unwrap(),
        2
    );

    let rejected_shorts = flags
        .select_shorts(&db, &FlagQuery::has(Flag::Rejected))
        .unwrap();
    let non_rejected_shorts = flags
        .select_shorts(&db, &FlagQuery::not(Flag::Rejected))
        .unwrap();
    assert_eq!(rejected_shorts.len(), 1);
    assert_eq!(non_rejected_shorts.len(), 2);
    assert!(!non_rejected_shorts.contains(&rejected_shorts[0]));
}

#[test]
fn stale_auth_events_are_rejected_against_state_before() {
    use mtxdb::room_flag::*;
    use mtxdb::timeline::*;
    use mtxdb::ShardType;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();

    let room_id = "!staleauth:test.conduwuit.local";
    let jsonl = format!(
        concat!(
            r#"{{"format":"mtxdb-export/v1","room_id":"{room}","server":"test.conduwuit.local","timeline":true,"t":"header"}}"#,
            "\n",
            r#"{{"event_id":"$create:test","depth":1,"stream_ordering":1,"auth_events":[],"prev_events":[],"outlier":false,"pdu":{{"auth_events":[],"content":{{"room_version":"11"}},"depth":1,"event_id":"$create:test","origin_server_ts":1000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","state_key":"","type":"m.room.create"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$join:test","depth":2,"stream_ordering":2,"auth_events":["$create:test"],"prev_events":["$create:test"],"outlier":false,"pdu":{{"auth_events":["$create:test"],"content":{{"membership":"join"}},"depth":2,"event_id":"$join:test","origin_server_ts":2000,"prev_events":["$create:test"],"room_id":"{room}","sender":"@alice:test","state_key":"@alice:test","type":"m.room.member"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$leave:test","depth":3,"stream_ordering":3,"auth_events":["$create:test","$join:test"],"prev_events":["$join:test"],"outlier":false,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{"membership":"leave"}},"depth":3,"event_id":"$leave:test","origin_server_ts":3000,"prev_events":["$join:test"],"room_id":"{room}","sender":"@alice:test","state_key":"@alice:test","type":"m.room.member"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$stale:test","depth":4,"stream_ordering":4,"auth_events":["$create:test","$join:test"],"prev_events":["$leave:test"],"outlier":false,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{}},"depth":4,"event_id":"$stale:test","origin_server_ts":4000,"prev_events":["$leave:test"],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"events":4,"t":"end"}}"#,
            "\n"
        ),
        room = room_id
    );

    let run = import_inputs(&db, &root, &[("room.jsonl", &jsonl)]);
    assert_eq!(
        (run.failures, run.all_exports, run.exported_events),
        (0, true, 4)
    );
    assert_eq!(run.rejection_counts.values().sum::<usize>(), 1);

    let flags = RoomFlagIndex::new(room_id.as_bytes());
    assert_eq!(flags.count(&db, &FlagQuery::has(Flag::Present)).unwrap(), 4);
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Rejected)).unwrap(),
        1,
        "the event is rejected against the state before it, not its cited auth events"
    );
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Timeline)).unwrap(),
        3
    );

    let page = TimelineIndex::open(db.pool(ShardType::EventDag).as_ref(), room_id)
        .page(&room_scope_id(room_id.as_bytes()), None, true, 100)
        .unwrap();
    let streams: Vec<i64> = page
        .entries
        .iter()
        .map(|entry| entry.stream_ordering)
        .collect();
    assert_eq!(
        streams,
        vec![1, 2, 3],
        "the rejected timeline event has no timeline slot"
    );
}

#[test]
fn rejected_state_event_does_not_change_later_state() {
    use mtxdb::export_schema::RejectionCode;
    use mtxdb::room_flag::*;
    use mtxdb::timeline::*;
    use mtxdb::ShardType;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();

    let room_id = "!staterollback:test.conduwuit.local";
    // $rules_ok makes the room public while Alice is joined. $rules_bad is
    // the same value but cites the stale join after Alice left, so only
    // check 5 rejects it. Bob's join passes checks 4 and 5 against the public
    // state, but cites $rules_bad, so only the rejected-auth cascade rejects
    // him.
    let jsonl = format!(
        concat!(
            r#"{{"format":"mtxdb-export/v1","room_id":"{room}","server":"test.conduwuit.local","timeline":true,"t":"header"}}"#,
            "\n",
            r#"{{"event_id":"$create:test","depth":1,"stream_ordering":1,"auth_events":[],"prev_events":[],"outlier":false,"pdu":{{"auth_events":[],"content":{{"room_version":"11"}},"depth":1,"event_id":"$create:test","origin_server_ts":1000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","state_key":"","type":"m.room.create"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$join:test","depth":2,"stream_ordering":2,"auth_events":["$create:test"],"prev_events":["$create:test"],"outlier":false,"pdu":{{"auth_events":["$create:test"],"content":{{"membership":"join"}},"depth":2,"event_id":"$join:test","origin_server_ts":2000,"prev_events":["$create:test"],"room_id":"{room}","sender":"@alice:test","state_key":"@alice:test","type":"m.room.member"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$rules_ok:test","depth":3,"stream_ordering":3,"auth_events":["$create:test","$join:test"],"prev_events":["$join:test"],"outlier":false,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{"join_rule":"public"}},"depth":3,"event_id":"$rules_ok:test","origin_server_ts":3000,"prev_events":["$join:test"],"room_id":"{room}","sender":"@alice:test","state_key":"","type":"m.room.join_rules"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$leave:test","depth":4,"stream_ordering":4,"auth_events":["$create:test","$join:test"],"prev_events":["$rules_ok:test"],"outlier":false,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{"membership":"leave"}},"depth":4,"event_id":"$leave:test","origin_server_ts":4000,"prev_events":["$rules_ok:test"],"room_id":"{room}","sender":"@alice:test","state_key":"@alice:test","type":"m.room.member"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$rules_bad:test","depth":5,"stream_ordering":5,"auth_events":["$create:test","$join:test"],"prev_events":["$leave:test"],"outlier":false,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{"join_rule":"public"}},"depth":5,"event_id":"$rules_bad:test","origin_server_ts":5000,"prev_events":["$leave:test"],"room_id":"{room}","sender":"@alice:test","state_key":"","type":"m.room.join_rules"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$bob:test","depth":6,"stream_ordering":6,"auth_events":["$create:test","$rules_bad:test"],"prev_events":["$rules_bad:test"],"outlier":false,"pdu":{{"auth_events":["$create:test","$rules_bad:test"],"content":{{"membership":"join"}},"depth":6,"event_id":"$bob:test","origin_server_ts":6000,"prev_events":["$rules_bad:test"],"room_id":"{room}","sender":"@bob:test","state_key":"@bob:test","type":"m.room.member"}},"t":"event"}}"#,
            "\n",
            r#"{{"events":6,"t":"end"}}"#,
            "\n"
        ),
        room = room_id
    );

    let run = import_inputs(&db, &root, &[("room.jsonl", &jsonl)]);
    assert_eq!(
        (run.failures, run.all_exports, run.exported_events),
        (0, true, 6)
    );
    assert_eq!(
        run.rejection_counts
            .get(&RejectionCode::AuthCheckFailed.to_u8())
            .copied(),
        Some(1),
        "$rules_bad fails check 5"
    );
    assert_eq!(
        run.rejection_counts
            .get(&RejectionCode::DependsOnRejectedAuthEvent.to_u8())
            .copied(),
        Some(1),
        "bob passes checks 4 and 5 but cites a rejected auth event"
    );

    let flags = RoomFlagIndex::new(room_id.as_bytes());
    assert_eq!(flags.count(&db, &FlagQuery::has(Flag::Present)).unwrap(), 6);
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Rejected)).unwrap(),
        2
    );
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Timeline)).unwrap(),
        4
    );

    let page = TimelineIndex::open(db.pool(ShardType::EventDag).as_ref(), room_id)
        .page(&room_scope_id(room_id.as_bytes()), None, true, 100)
        .unwrap();
    let streams: Vec<i64> = page
        .entries
        .iter()
        .map(|entry| entry.stream_ordering)
        .collect();
    assert_eq!(streams, vec![1, 2, 3, 4]);
}

#[test]
fn partial_base_state_is_left_unchecked() {
    use mtxdb::room_flag::*;
    use mtxdb::timeline::*;
    use mtxdb::ShardType;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();

    let room_id = "!partialstate:test.conduwuit.local";
    // $partial's second parent is missing, so its base state is only the
    // leave. Checking against that partial state would reject Alice; with an
    // incomplete base the event must stay unchecked instead.
    // $child descends from $partial. Its own parents are all present, but
    // because it inherits an incomplete base it must also stay unchecked.
    let jsonl = format!(
        concat!(
            r#"{{"format":"mtxdb-export/v1","room_id":"{room}","server":"test.conduwuit.local","timeline":true,"t":"header"}}"#,
            "\n",
            r#"{{"event_id":"$create:test","depth":1,"stream_ordering":1,"auth_events":[],"prev_events":[],"outlier":false,"pdu":{{"auth_events":[],"content":{{"room_version":"11"}},"depth":1,"event_id":"$create:test","origin_server_ts":1000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","state_key":"","type":"m.room.create"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$join:test","depth":2,"stream_ordering":2,"auth_events":["$create:test"],"prev_events":["$create:test"],"outlier":false,"pdu":{{"auth_events":["$create:test"],"content":{{"membership":"join"}},"depth":2,"event_id":"$join:test","origin_server_ts":2000,"prev_events":["$create:test"],"room_id":"{room}","sender":"@alice:test","state_key":"@alice:test","type":"m.room.member"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$leave:test","depth":3,"stream_ordering":3,"auth_events":["$create:test","$join:test"],"prev_events":["$join:test"],"outlier":false,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{"membership":"leave"}},"depth":3,"event_id":"$leave:test","origin_server_ts":3000,"prev_events":["$join:test"],"room_id":"{room}","sender":"@alice:test","state_key":"@alice:test","type":"m.room.member"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$partial:test","depth":4,"stream_ordering":4,"auth_events":["$create:test","$join:test"],"prev_events":["$leave:test","$missing:test"],"outlier":false,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{}},"depth":4,"event_id":"$partial:test","origin_server_ts":4000,"prev_events":["$leave:test","$missing:test"],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$child:test","depth":5,"stream_ordering":5,"auth_events":["$create:test","$join:test"],"prev_events":["$partial:test"],"outlier":false,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{}},"depth":5,"event_id":"$child:test","origin_server_ts":5000,"prev_events":["$partial:test"],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"events":5,"t":"end"}}"#,
            "\n"
        ),
        room = room_id
    );

    let run = import_inputs(&db, &root, &[("room.jsonl", &jsonl)]);
    assert_eq!(
        (run.failures, run.all_exports, run.exported_events),
        (0, true, 5)
    );
    assert_eq!(
        run.rejection_counts.values().sum::<usize>(),
        0,
        "a missing parent and its descendants stay unchecked"
    );

    let flags = RoomFlagIndex::new(room_id.as_bytes());
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Rejected)).unwrap(),
        0
    );
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Timeline)).unwrap(),
        5
    );

    let page = TimelineIndex::open(db.pool(ShardType::EventDag).as_ref(), room_id)
        .page(&room_scope_id(room_id.as_bytes()), None, true, 100)
        .unwrap();
    let streams: Vec<i64> = page
        .entries
        .iter()
        .map(|entry| entry.stream_ordering)
        .collect();
    assert_eq!(streams, vec![1, 2, 3, 4, 5]);
}

#[test]
fn base_is_incomplete_inherits_from_parents() {
    let instance = [0_u8; STATE_GROUP_ID_LENGTH];
    let complete = SharedState::new(StateSet::new(), instance, true);
    let incomplete = SharedState::new(StateSet::new(), instance, false);

    assert!(!base_is_incomplete(false, std::slice::from_ref(&complete)));
    assert!(base_is_incomplete(true, std::slice::from_ref(&complete)));
    assert!(base_is_incomplete(false, std::slice::from_ref(&incomplete)));
}

#[test]
fn incomplete_base_is_inherited_by_descendants() {
    use mtxdb::room_flag::*;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();

    let room_id = "!inheritedgap:test.conduwuit.local";
    // $orphan_leave has no prev_events, so its own base is incomplete even
    // though it is walked. Its leave state is non-empty, so a child judged
    // against it would reject Alice. The child inherits the gap instead.
    let jsonl = format!(
        concat!(
            r#"{{"format":"mtxdb-export/v1","room_id":"{room}","server":"test.conduwuit.local","timeline":true,"t":"header"}}"#,
            "\n",
            r#"{{"event_id":"$create:test","depth":1,"stream_ordering":1,"auth_events":[],"prev_events":[],"outlier":false,"pdu":{{"auth_events":[],"content":{{"room_version":"11"}},"depth":1,"event_id":"$create:test","origin_server_ts":1000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","state_key":"","type":"m.room.create"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$join:test","depth":2,"stream_ordering":2,"auth_events":["$create:test"],"prev_events":["$create:test"],"outlier":false,"pdu":{{"auth_events":["$create:test"],"content":{{"membership":"join"}},"depth":2,"event_id":"$join:test","origin_server_ts":2000,"prev_events":["$create:test"],"room_id":"{room}","sender":"@alice:test","state_key":"@alice:test","type":"m.room.member"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$orphan_leave:test","depth":3,"stream_ordering":3,"auth_events":["$create:test","$join:test"],"prev_events":[],"outlier":false,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{"membership":"leave"}},"depth":3,"event_id":"$orphan_leave:test","origin_server_ts":3000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","state_key":"@alice:test","type":"m.room.member"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$child:test","depth":4,"stream_ordering":4,"auth_events":["$create:test","$join:test"],"prev_events":["$orphan_leave:test"],"outlier":false,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{}},"depth":4,"event_id":"$child:test","origin_server_ts":4000,"prev_events":["$orphan_leave:test"],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"events":4,"t":"end"}}"#,
            "\n"
        ),
        room = room_id
    );

    let run = import_inputs(&db, &root, &[("room.jsonl", &jsonl)]);
    assert_eq!(
        (run.failures, run.all_exports, run.exported_events),
        (0, true, 4)
    );
    assert_eq!(
        run.rejection_counts.values().sum::<usize>(),
        0,
        "the child inherits $orphan_leave's incomplete base"
    );

    let flags = RoomFlagIndex::new(room_id.as_bytes());
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Rejected)).unwrap(),
        0
    );
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Timeline)).unwrap(),
        4
    );
}

#[test]
fn duplicate_cited_auth_events_are_still_rejected() {
    use mtxdb::export_schema::RejectionCode;
    use mtxdb::room_flag::*;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();

    let room_id = "!dupauth:test.conduwuit.local";
    let jsonl = format!(
        concat!(
            r#"{{"format":"mtxdb-export/v1","room_id":"{room}","server":"test.conduwuit.local","timeline":true,"t":"header"}}"#,
            "\n",
            r#"{{"event_id":"$create:test","depth":1,"stream_ordering":1,"auth_events":[],"prev_events":[],"outlier":false,"pdu":{{"auth_events":[],"content":{{"room_version":"11"}},"depth":1,"event_id":"$create:test","origin_server_ts":1000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","state_key":"","type":"m.room.create"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$join:test","depth":2,"stream_ordering":2,"auth_events":["$create:test"],"prev_events":["$create:test"],"outlier":false,"pdu":{{"auth_events":["$create:test"],"content":{{"membership":"join"}},"depth":2,"event_id":"$join:test","origin_server_ts":2000,"prev_events":["$create:test"],"room_id":"{room}","sender":"@alice:test","state_key":"@alice:test","type":"m.room.member"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$join2:test","depth":3,"stream_ordering":3,"auth_events":["$create:test","$join:test"],"prev_events":["$join:test"],"outlier":false,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{"membership":"join"}},"depth":3,"event_id":"$join2:test","origin_server_ts":3000,"prev_events":["$join:test"],"room_id":"{room}","sender":"@alice:test","state_key":"@alice:test","type":"m.room.member"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$dup:test","depth":4,"stream_ordering":4,"auth_events":["$create:test","$join:test","$join2:test"],"prev_events":["$join2:test"],"outlier":false,"pdu":{{"auth_events":["$create:test","$join:test","$join2:test"],"content":{{}},"depth":4,"event_id":"$dup:test","origin_server_ts":4000,"prev_events":["$join2:test"],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"events":4,"t":"end"}}"#,
            "\n"
        ),
        room = room_id
    );

    let run = import_inputs(&db, &root, &[("room.jsonl", &jsonl)]);
    assert_eq!(
        (run.failures, run.all_exports, run.exported_events),
        (0, true, 4)
    );
    assert_eq!(
        run.rejection_counts
            .get(&RejectionCode::DuplicateAuthEventKey.to_u8())
            .copied(),
        Some(1),
        "check 4 still rejects duplicate cited auth events"
    );

    let flags = RoomFlagIndex::new(room_id.as_bytes());
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Rejected)).unwrap(),
        1
    );
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Timeline)).unwrap(),
        3
    );
}

#[test]
fn exported_rejected_flag_is_not_trusted_for_state() {
    use mtxdb::room_flag::*;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();

    let room_id = "!exporterflag:test.conduwuit.local";
    let jsonl = format!(
        concat!(
            r#"{{"format":"mtxdb-export/v1","room_id":"{room}","server":"test.conduwuit.local","timeline":true,"t":"header"}}"#,
            "\n",
            r#"{{"event_id":"$create:test","depth":1,"stream_ordering":1,"auth_events":[],"prev_events":[],"outlier":false,"pdu":{{"auth_events":[],"content":{{"room_version":"11"}},"depth":1,"event_id":"$create:test","origin_server_ts":1000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","state_key":"","type":"m.room.create"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$join:test","depth":2,"stream_ordering":2,"auth_events":["$create:test"],"prev_events":["$create:test"],"outlier":false,"pdu":{{"auth_events":["$create:test"],"content":{{"membership":"join"}},"depth":2,"event_id":"$join:test","origin_server_ts":2000,"prev_events":["$create:test"],"room_id":"{room}","sender":"@alice:test","state_key":"@alice:test","type":"m.room.member"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$rules:test","depth":3,"stream_ordering":3,"rejected":true,"auth_events":["$create:test","$join:test"],"prev_events":["$join:test"],"outlier":false,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{"join_rule":"public"}},"depth":3,"event_id":"$rules:test","origin_server_ts":3000,"prev_events":["$join:test"],"room_id":"{room}","sender":"@alice:test","state_key":"","type":"m.room.join_rules"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$bob:test","depth":4,"stream_ordering":4,"auth_events":["$create:test","$rules:test"],"prev_events":["$rules:test"],"outlier":false,"pdu":{{"auth_events":["$create:test","$rules:test"],"content":{{"membership":"join"}},"depth":4,"event_id":"$bob:test","origin_server_ts":4000,"prev_events":["$rules:test"],"room_id":"{room}","sender":"@bob:test","state_key":"@bob:test","type":"m.room.member"}},"t":"event"}}"#,
            "\n",
            r#"{{"events":4,"t":"end"}}"#,
            "\n"
        ),
        room = room_id
    );

    let run = import_inputs(&db, &root, &[("room.jsonl", &jsonl)]);
    assert_eq!(
        (run.failures, run.all_exports, run.exported_events),
        (0, true, 4)
    );
    assert_eq!(
        run.rejection_counts.values().sum::<usize>(),
        0,
        "the export's rejected flag does not reject a valid event"
    );

    let flags = RoomFlagIndex::new(room_id.as_bytes());
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Rejected)).unwrap(),
        0
    );
    assert_eq!(flags.count(&db, &FlagQuery::has(Flag::Present)).unwrap(), 4);
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Timeline)).unwrap(),
        4
    );
}

#[test]
fn flag_index_timeline_count_matches_timeline_index() {
    use crate::interchange::parse_room;
    use mtxdb::room_flag::*;
    use mtxdb::timeline::*;
    use mtxdb::ShardType;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();

    let room_id = "!timeline_match:test.conduwuit.local";
    let jsonl = format!(
        concat!(
            r#"{{"format":"mtxdb-export/v1","room_id":"{room}","server":"test.conduwuit.local","timeline":true,"t":"header"}}"#,
            "\n",
            r#"{{"event_id":"$backfill:test","depth":10,"stream_ordering":-5,"auth_events":[],"prev_events":[],"outlier":false,"pdu":{{"auth_events":[],"content":{{"room_version":"11"}},"depth":10,"event_id":"$backfill:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":1000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","state_key":"","type":"m.room.create"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$normal:test","depth":10,"stream_ordering":100,"auth_events":[],"prev_events":[],"outlier":false,"pdu":{{"auth_events":[],"content":{{}},"depth":10,"event_id":"$normal:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":2000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$later:test","depth":11,"stream_ordering":101,"auth_events":[],"prev_events":[],"outlier":false,"pdu":{{"auth_events":[],"content":{{}},"depth":11,"event_id":"$later:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":3000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$notimeline:test","depth":12,"auth_events":[],"prev_events":[],"outlier":false,"pdu":{{"auth_events":[],"content":{{}},"depth":12,"event_id":"$notimeline:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":4000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$outlier:test","depth":13,"stream_ordering":200,"auth_events":[],"prev_events":[],"outlier":true,"pdu":{{"auth_events":[],"content":{{}},"depth":13,"event_id":"$outlier:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":5000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"events":5,"t":"end"}}"#,
            "\n"
        ),
        room = room_id
    );

    let export = parse_room(jsonl.as_bytes()).unwrap();
    let (written, _) = export_import::write_room_with_findings(&db, &export, 64 * 1024).unwrap();
    assert_eq!(written.events, 5);

    let room_scope = room_scope_id(room_id.as_bytes());
    let timeline_index = TimelineIndex::open(db.pool(ShardType::EventDag).as_ref(), room_id);
    let page = timeline_index.page(&room_scope, None, true, 100).unwrap();

    let flags = RoomFlagIndex::new(room_id.as_bytes());
    let timeline_count = flags.count(&db, &FlagQuery::has(Flag::Timeline)).unwrap();
    assert_eq!(timeline_count, page.entries.len() as u64);
    assert_eq!(timeline_count, 1);
}

#[test]
fn legacy_export_soft_failed_outlier_stays_out_of_timeline() {
    use crate::interchange::parse_room;
    use mtxdb::room_flag::*;
    use mtxdb::timeline::*;
    use mtxdb::ShardType;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();

    let room_id = "!legacy:test.conduwuit.local";
    // Exactly what the exporter wrote before stream orderings: no `timeline`
    // header, no `stream_ordering` or `arrival_ordering`, and only the
    // outlier/soft-failed flags. Timeline entries therefore come from the
    // legacy origin_server_ts fallback.
    let jsonl = format!(
        concat!(
            r#"{{"format":"mtxdb-export/v1","room_id":"{room}","server":"test.conduwuit.local","t":"header"}}"#,
            "\n",
            r#"{{"event_id":"$create:test","auth_events":[],"prev_events":[],"outlier":false,"soft_failed":false,"rejected":false,"pdu":{{"auth_events":[],"content":{{"room_version":"11"}},"depth":1,"event_id":"$create:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":1000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","state_key":"","type":"m.room.create"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$join:test","auth_events":["$create:test"],"prev_events":["$create:test"],"outlier":false,"soft_failed":false,"rejected":false,"pdu":{{"auth_events":["$create:test"],"content":{{"membership":"join"}},"depth":2,"event_id":"$join:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":1500,"prev_events":["$create:test"],"room_id":"{room}","sender":"@alice:test","state_key":"@alice:test","type":"m.room.member"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$live:test","auth_events":["$create:test","$join:test"],"prev_events":["$join:test"],"outlier":false,"soft_failed":false,"rejected":false,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{}},"depth":3,"event_id":"$live:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":2000,"prev_events":["$join:test"],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$soft:test","auth_events":["$create:test","$join:test"],"prev_events":["$join:test"],"outlier":true,"soft_failed":true,"soft_fail_code":1,"rejected":false,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{}},"depth":4,"event_id":"$soft:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":3000,"prev_events":["$join:test"],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"events":4,"t":"end"}}"#,
            "\n"
        ),
        room = room_id
    );

    let export = parse_room(jsonl.as_bytes()).unwrap();
    assert!(
        !export.timeline,
        "legacy exports carry no timeline contract"
    );
    assert_eq!(export.events[3].stream_ordering, None);
    assert_eq!(export.events[3].arrival_ordering, None);
    let (written, _) = export_import::write_room_with_findings(&db, &export, 64 * 1024).unwrap();
    assert_eq!(written.events, 4);

    let room_scope = room_scope_id(room_id.as_bytes());
    let timeline_index = TimelineIndex::open(db.pool(ShardType::EventDag).as_ref(), room_id);
    let page = timeline_index.page(&room_scope, None, true, 100).unwrap();

    // The two non-outliers make it in; the soft-failed outlier stays out even
    // though its PDU carries a depth and timestamp.
    let streams: Vec<i64> = page.entries.iter().map(|e| e.stream_ordering).collect();
    assert_eq!(streams, vec![1000, 1500, 2000]);

    let flags = RoomFlagIndex::new(room_id.as_bytes());
    assert_eq!(flags.count(&db, &FlagQuery::has(Flag::Present)).unwrap(), 4);
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Timeline)).unwrap(),
        3
    );
    let non_timeline = flags
        .select_shorts(
            &db,
            &FlagQuery::and([
                FlagQuery::has(Flag::Present),
                FlagQuery::not(Flag::Timeline),
            ]),
        )
        .unwrap();
    assert_eq!(
        non_timeline.len(),
        1,
        "only the soft-failed event is hidden"
    );
}

#[test]
fn stream_and_arrival_order_survive_import() {
    use crate::interchange::parse_room;
    use mtxdb::interop::*;
    use mtxdb::room_flag::*;
    use mtxdb::timeline::*;
    use mtxdb::ShardType;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();

    let room_id = "!nostream:test.conduwuit.local";
    // origin_server_ts values are deliberately unrelated to the orderings, so
    // any fallback to them would be visible in the timeline.
    let jsonl = format!(
        concat!(
            r#"{{"format":"mtxdb-export/v1","room_id":"{room}","server":"test.conduwuit.local","timeline":true,"t":"header"}}"#,
            "\n",
            r#"{{"event_id":"$create:test","depth":1,"stream_ordering":1,"auth_events":[],"prev_events":[],"outlier":false,"pdu":{{"auth_events":[],"content":{{"room_version":"11"}},"depth":1,"event_id":"$create:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":900000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","state_key":"","type":"m.room.create"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$join:test","depth":2,"stream_ordering":2,"auth_events":["$create:test"],"prev_events":["$create:test"],"outlier":false,"pdu":{{"auth_events":["$create:test"],"content":{{"membership":"join"}},"depth":2,"event_id":"$join:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":850000,"prev_events":["$create:test"],"room_id":"{room}","sender":"@alice:test","state_key":"@alice:test","type":"m.room.member"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$live:test","depth":3,"stream_ordering":3,"auth_events":["$create:test","$join:test"],"prev_events":["$join:test"],"outlier":false,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{}},"depth":3,"event_id":"$live:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":800000,"prev_events":["$join:test"],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$backfilled:test","depth":5,"stream_ordering":-5,"auth_events":["$create:test","$join:test"],"prev_events":["$live:test"],"outlier":false,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{}},"depth":5,"event_id":"$backfilled:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":700000,"prev_events":["$live:test"],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$soft:test","depth":9,"arrival_ordering":7,"auth_events":["$create:test","$join:test"],"prev_events":["$backfilled:test"],"outlier":true,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{}},"depth":9,"event_id":"$soft:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":600000,"prev_events":["$backfilled:test"],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"events":5,"t":"end"}}"#,
            "\n"
        ),
        room = room_id
    );

    let export = parse_room(jsonl.as_bytes()).unwrap();
    assert_eq!(export.events[4].arrival_ordering, Some(7));
    let (written, _) = export_import::write_room_with_findings(&db, &export, 64 * 1024).unwrap();
    assert_eq!(written.events, 5);

    let room_scope = room_scope_id(room_id.as_bytes());
    let timeline_index = TimelineIndex::open(db.pool(ShardType::EventDag).as_ref(), room_id);
    let page = timeline_index.page(&room_scope, None, true, 100).unwrap();

    // Live and backfilled positions are preserved exactly; the soft-failed
    // event has no timeline position and stays out.
    let streams: Vec<i64> = page.entries.iter().map(|e| e.stream_ordering).collect();
    assert_eq!(streams, vec![1, 2, 3, -5]);
    let depths: Vec<i64> = page
        .entries
        .iter()
        .map(|e| e.topological_ordering)
        .collect();
    assert_eq!(depths, vec![1, 2, 3, 5]);

    let flags = RoomFlagIndex::new(room_id.as_bytes());
    assert_eq!(flags.count(&db, &FlagQuery::has(Flag::Present)).unwrap(), 5);
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Timeline)).unwrap(),
        4
    );

    // Events with Present AND NOT Timeline (kept out of timeline)
    let non_timeline_shorts = flags
        .select_shorts(
            &db,
            &FlagQuery::and([
                FlagQuery::has(Flag::Present),
                FlagQuery::not(Flag::Timeline),
            ]),
        )
        .unwrap();
    assert_eq!(non_timeline_shorts.len(), 1);

    // The arrival ordering is a source-recorded fact and is kept in the
    // verdict. This export does not mark the event soft-failed.
    let record = event_record_id(b"$soft:test");
    let bytes = stored(
        &db,
        EVENT_POOL,
        event_verdict_collection_id(&record),
        record,
    )
    .expect("soft-failed event has a verdict record");
    let (_, verdict) = decode_verdict(&bytes).unwrap();
    assert_eq!(verdict.arrival_ordering, Some(7));
    assert_eq!(
        verdict.soft_fail, None,
        "this export does not mark it soft-failed"
    );
}

#[test]
fn imported_soft_fail_is_queryable_and_stays_out_of_timeline() {
    use crate::interchange::parse_room;
    use mtxdb::interop::*;
    use mtxdb::room_flag::*;
    use mtxdb::timeline::*;
    use mtxdb::ShardType;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();

    let room_id = "!importedsoft:test.conduwuit.local";
    let jsonl = format!(
        concat!(
            r#"{{"format":"mtxdb-export/v1","room_id":"{room}","server":"test.conduwuit.local","timeline":true,"t":"header"}}"#,
            "\n",
            r#"{{"event_id":"$create:test","depth":1,"stream_ordering":1,"auth_events":[],"prev_events":[],"outlier":false,"pdu":{{"auth_events":[],"content":{{"room_version":"11"}},"depth":1,"event_id":"$create:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":1000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","state_key":"","type":"m.room.create"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$join:test","depth":2,"stream_ordering":2,"auth_events":["$create:test"],"prev_events":["$create:test"],"outlier":false,"pdu":{{"auth_events":["$create:test"],"content":{{"membership":"join"}},"depth":2,"event_id":"$join:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":2000,"prev_events":["$create:test"],"room_id":"{room}","sender":"@alice:test","state_key":"@alice:test","type":"m.room.member"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$live:test","depth":3,"stream_ordering":3,"auth_events":["$create:test","$join:test"],"prev_events":["$join:test"],"outlier":false,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{}},"depth":3,"event_id":"$live:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":3000,"prev_events":["$join:test"],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$soft:test","depth":4,"arrival_ordering":7,"soft_failed":true,"soft_fail_code":2,"auth_events":["$create:test","$join:test"],"prev_events":["$live:test"],"outlier":true,"pdu":{{"auth_events":["$create:test","$join:test"],"content":{{}},"depth":4,"event_id":"$soft:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":4000,"prev_events":["$live:test"],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"events":4,"t":"end"}}"#,
            "\n"
        ),
        room = room_id
    );

    let export = parse_room(jsonl.as_bytes()).unwrap();
    let (written, findings) =
        export_import::write_room_with_findings(&db, &export, 64 * 1024).unwrap();
    assert_eq!(written.events, 4);
    assert_eq!(findings.rejection_counts.values().sum::<usize>(), 0);

    let flags = RoomFlagIndex::new(room_id.as_bytes());
    assert_eq!(flags.count(&db, &FlagQuery::has(Flag::Present)).unwrap(), 4);
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Timeline)).unwrap(),
        3
    );
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::SoftFailed)).unwrap(),
        1
    );
    assert_eq!(
        flags
            .select_shorts(&db, &FlagQuery::has(Flag::SoftFailed))
            .unwrap()
            .len(),
        1,
        "--flag soft-failed selects the imported event"
    );

    let page = TimelineIndex::open(db.pool(ShardType::EventDag).as_ref(), room_id)
        .page(&room_scope_id(room_id.as_bytes()), None, true, 100)
        .unwrap();
    let streams: Vec<i64> = page.entries.iter().map(|e| e.stream_ordering).collect();
    assert_eq!(
        streams,
        vec![1, 2, 3],
        "the soft-failed event has no timeline slot"
    );

    let record = event_record_id(b"$soft:test");
    let bytes = stored(
        &db,
        EVENT_POOL,
        event_verdict_collection_id(&record),
        record,
    )
    .expect("soft-failed event has a verdict record");
    let (_, verdict) = decode_verdict(&bytes).unwrap();
    assert_eq!(verdict.soft_fail, Some(2), "the exported reason survives");
    assert_eq!(verdict.arrival_ordering, Some(7));
}

#[test]
fn negative_arrival_ordering_is_rejected_on_import() {
    use crate::interchange::parse_room;

    let room_id = "!badarrival:test.conduwuit.local";
    let jsonl = format!(
        concat!(
            r#"{{"format":"mtxdb-export/v1","room_id":"{room}","server":"test.conduwuit.local","timeline":true,"t":"header"}}"#,
            "\n",
            r#"{{"event_id":"$create:test","stream_ordering":1,"auth_events":[],"prev_events":[],"outlier":false,"pdu":{{"auth_events":[],"content":{{"room_version":"11"}},"depth":1,"event_id":"$create:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":1000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","state_key":"","type":"m.room.create"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$soft:test","arrival_ordering":-1,"auth_events":[],"prev_events":[],"outlier":true,"pdu":{{"auth_events":[],"content":{{}},"depth":2,"event_id":"$soft:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":2000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"events":2,"t":"end"}}"#,
            "\n"
        ),
        room = room_id
    );

    let error = parse_room(jsonl.as_bytes()).unwrap_err().to_string();
    assert!(error.contains("arrival_ordering"), "{error}");
    assert!(error.contains("non-negative"), "{error}");
}

#[test]
fn rebuild_fidelity_reproduces_exact_bitmaps_after_wipe() {
    use crate::interchange::parse_room;
    use mtxdb::room_flag::*;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let export = parse_room(ROOM_EXPORT_FIXTURE.as_bytes()).unwrap();
    let (written, _) = export_import::write_room_with_findings(&db, &export, 64 * 1024).unwrap();
    assert_eq!(written.events, 5);

    let flags = RoomFlagIndex::new(FIXTURE_ROOM.as_bytes());
    let present_before = flags.count(&db, &FlagQuery::has(Flag::Present)).unwrap();
    let timeline_before = flags.count(&db, &FlagQuery::has(Flag::Timeline)).unwrap();
    let outlier_before = flags.count(&db, &FlagQuery::has(Flag::Outlier)).unwrap();
    let rejected_before = flags.count(&db, &FlagQuery::has(Flag::Rejected)).unwrap();
    let soft_failed_before = flags.count(&db, &FlagQuery::has(Flag::SoftFailed)).unwrap();
    let redacted_before = flags.count(&db, &FlagQuery::has(Flag::Redacted)).unwrap();

    let shorts_present_before = flags
        .select_shorts(&db, &FlagQuery::has(Flag::Present))
        .unwrap();
    let shorts_timeline_before = flags
        .select_shorts(&db, &FlagQuery::has(Flag::Timeline))
        .unwrap();

    // Wipe the index: remove both its metadata collection and the active
    // chunk generation, so the rebuild below starts from nothing.
    let generation = flags.meta(&db).unwrap().generation;
    let wipe = db.begin_transaction();
    wipe.delete_collection(flags.pool(), *flags.collection())
        .unwrap();
    wipe.delete_collection(flags.pool(), flags.chunk_collection(generation))
        .unwrap();
    wipe.commit().unwrap();

    // Run rebuild
    let count = flags.rebuild(&db).unwrap();
    assert_eq!(count, 5);

    // Assert exact equality
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Present)).unwrap(),
        present_before
    );
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Timeline)).unwrap(),
        timeline_before
    );
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Outlier)).unwrap(),
        outlier_before
    );
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Rejected)).unwrap(),
        rejected_before
    );
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::SoftFailed)).unwrap(),
        soft_failed_before
    );
    assert_eq!(
        flags.count(&db, &FlagQuery::has(Flag::Redacted)).unwrap(),
        redacted_before
    );

    assert_eq!(
        flags
            .select_shorts(&db, &FlagQuery::has(Flag::Present))
            .unwrap(),
        shorts_present_before
    );
    assert_eq!(
        flags
            .select_shorts(&db, &FlagQuery::has(Flag::Timeline))
            .unwrap(),
        shorts_timeline_before
    );
}

#[test]
fn timeline_ordering_respects_depth_signed_stream_and_timeline_flag() {
    use crate::interchange::parse_room;
    use mtxdb::timeline::*;
    use mtxdb::ShardType;

    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();

    let room_id = "!test_ordering:test.conduwuit.local";
    let jsonl = format!(
        concat!(
            r#"{{"format":"mtxdb-export/v1","room_id":"{room}","server":"test.conduwuit.local","timeline":true,"t":"header"}}"#,
            "\n",
            r#"{{"event_id":"$backfill:test","depth":10,"stream_ordering":-5,"auth_events":[],"prev_events":[],"outlier":false,"pdu":{{"auth_events":[],"content":{{"room_version":"11"}},"depth":10,"event_id":"$backfill:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":1000,"prev_events":[],"room_id":"{room}","sender":"@alice:test","state_key":"","type":"m.room.create"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$join:test","depth":10,"stream_ordering":0,"auth_events":["$backfill:test"],"prev_events":["$backfill:test"],"outlier":false,"pdu":{{"auth_events":["$backfill:test"],"content":{{"membership":"join"}},"depth":10,"event_id":"$join:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":1500,"prev_events":["$backfill:test"],"room_id":"{room}","sender":"@alice:test","state_key":"@alice:test","type":"m.room.member"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$normal:test","depth":10,"stream_ordering":100,"auth_events":["$backfill:test","$join:test"],"prev_events":["$join:test"],"outlier":false,"pdu":{{"auth_events":["$backfill:test","$join:test"],"content":{{}},"depth":10,"event_id":"$normal:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":2000,"prev_events":["$join:test"],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$later:test","depth":11,"stream_ordering":101,"auth_events":["$backfill:test","$join:test"],"prev_events":["$normal:test"],"outlier":false,"pdu":{{"auth_events":["$backfill:test","$join:test"],"content":{{}},"depth":11,"event_id":"$later:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":3000,"prev_events":["$normal:test"],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$notimeline:test","depth":12,"auth_events":["$backfill:test","$join:test"],"prev_events":["$later:test"],"outlier":false,"pdu":{{"auth_events":["$backfill:test","$join:test"],"content":{{}},"depth":12,"event_id":"$notimeline:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":4000,"prev_events":["$later:test"],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"event_id":"$outlier:test","depth":13,"stream_ordering":200,"auth_events":["$backfill:test","$join:test"],"prev_events":["$notimeline:test"],"outlier":true,"pdu":{{"auth_events":["$backfill:test","$join:test"],"content":{{}},"depth":13,"event_id":"$outlier:test","hashes":{{"sha256":"dummy"}},"origin_server_ts":5000,"prev_events":["$notimeline:test"],"room_id":"{room}","sender":"@alice:test","type":"m.room.message"}},"t":"event"}}"#,
            "\n",
            r#"{{"events":6,"t":"end"}}"#,
            "\n"
        ),
        room = room_id
    );

    let export = parse_room(jsonl.as_bytes()).unwrap();
    assert!(export.timeline);
    assert_eq!(export.events.len(), 6);

    let (written, _) = export_import::write_room_with_findings(&db, &export, 64 * 1024).unwrap();
    assert_eq!(written.events, 6);

    let room_scope = room_scope_id(room_id.as_bytes());
    let timeline_index = TimelineIndex::open(db.pool(ShardType::EventDag).as_ref(), room_id);
    let page = timeline_index.page(&room_scope, None, true, 10).unwrap();

    // Only the 4 timeline events should be present ($notimeline and $outlier excluded)
    assert_eq!(page.entries.len(), 4);

    // Verify ordering:
    // 1. $backfill (depth: 10, stream: -5)
    assert_eq!(page.entries[0].topological_ordering, 10);
    assert_eq!(page.entries[0].stream_ordering, -5);

    // 2. $join (depth: 10, stream: 0)
    assert_eq!(page.entries[1].topological_ordering, 10);
    assert_eq!(page.entries[1].stream_ordering, 0);

    // 3. $normal (depth: 10, stream: 100)
    assert_eq!(page.entries[2].topological_ordering, 10);
    assert_eq!(page.entries[2].stream_ordering, 100);

    // 4. $later (depth: 11, stream: 101)
    assert_eq!(page.entries[3].topological_ordering, 11);
    assert_eq!(page.entries[3].stream_ordering, 101);

    // Verify re-import idempotence: re-importing the same export does not duplicate entries
    let (written_again, _) =
        export_import::write_room_with_findings(&db, &export, 64 * 1024).unwrap();
    assert_eq!(written_again.events, 6);
    let page_after = timeline_index.page(&room_scope, None, true, 10).unwrap();
    assert_eq!(page_after.entries.len(), 4);
    assert_eq!(page.entries, page_after.entries);
}

#[test]
fn evict_database_pages_succeeds_on_valid_store() {
    let root = unique_temp_dir();
    let db = Database::open(root.clone()).unwrap();
    let export = crate::interchange::parse_room(ROOM_EXPORT_FIXTURE.as_bytes()).unwrap();
    let (written, _) = export_import::write_room_with_findings(&db, &export, 64 * 1024).unwrap();
    assert_eq!(written.events, 5);
    drop(db);

    let (files, bytes) = crate::cmd::evict_database_pages(&root).unwrap();
    assert!(files > 0);
    assert!(bytes > 0);
}
