use std::path::PathBuf;
use std::sync::Arc;

use super::*;

fn test_root(name: &str) -> PathBuf {
    let path =
        std::env::temp_dir().join(format!("mtxdb-server-keys-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn observed(key_id: &str, key: &[u8], valid_until_ts: u64) -> ObservedKey {
    ObservedKey {
        key_id: key_id.as_bytes().to_vec(),
        key: key.to_vec(),
        valid_until_ts,
        expired_ts: None,
    }
}

fn retired(key_id: &str, key: &[u8], valid_until_ts: u64, expired_ts: u64) -> ObservedKey {
    ObservedKey {
        expired_ts: Some(expired_ts),
        ..observed(key_id, key, valid_until_ts)
    }
}

const WINDOW: u64 = 1_000;

/// The kept keys of a merge; tests that care about issues call
/// [`update_server_keys`] directly.
fn merge(db: &Database, update: &KeyUpdate<'_>) -> Result<Vec<ServerKey>, StorageError> {
    update_server_keys(db, update).map(|merged| merged.keys)
}

fn update<'a>(
    origin: &'a str,
    sighting: KeySighting,
    source: &'a str,
    observed: &'a [ObservedKey],
    now_ms: u64,
) -> KeyUpdate<'a> {
    KeyUpdate {
        origin: origin.as_bytes(),
        sighting,
        source: source.as_bytes(),
        observed,
        now_ms,
        notary: None,
        promotion_window_ms: WINDOW,
        snapshot: false,
        limits: KeyLimits::default(),
    }
}

fn open(name: &str) -> (Database, PathBuf) {
    let root = test_root(name);
    (Database::open(root.clone()).unwrap(), root)
}

#[test]
fn key_record_round_trips_and_rejects_damage() {
    let key = ServerKey {
        key_id: b"ed25519:a".to_vec(),
        key: vec![7; 32],
        valid_until_ts: u64::MAX - 1,
        source: b"notary.example".to_vec(),
        ts_added_ms: 42,
        tier: KeyTier::Provisional,
        expired_ts: Some(7),
        promote_by_ms: 99,
        digest: key_digest(&[7; 32]),
        first_seen_ms: 5,
        last_active_ms: 6,
        corroborated: true,
        inferred_retired: true,
    };
    let bytes = encode_server_key_record(b"a.example", &key).unwrap();
    assert_eq!(
        decode_server_key_record(&bytes).unwrap(),
        (b"a.example".to_vec(), key)
    );

    assert!(decode_server_key_record(&[]).is_err());
    for cut in 0..bytes.len() {
        assert!(
            decode_server_key_record(&bytes[..cut]).is_err(),
            "cut at {cut}"
        );
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(decode_server_key_record(&trailing).is_err());
    let mut bad_tier = bytes.clone();
    bad_tier[1] = 9;
    let mut bad_flag = bytes.clone();
    bad_flag[2] = 0x80;
    assert!(decode_server_key_record(&bad_flag).is_err());
    assert!(decode_server_key_record(&bad_tier).is_err());
    let mut bad_version = bytes;
    bad_version[0] = 2;
    assert!(decode_server_key_record(&bad_version).is_err());
}

#[test]
fn first_key_seen_wins_and_tier_follows_the_sighting() {
    let (db, root) = open("first");
    let keys = [observed("ed25519:a", b"AAAA", 100)];
    let direct = merge(&db, &update("o", KeySighting::Direct, "origin", &keys, 1)).unwrap();
    assert_eq!(direct[0].tier, KeyTier::Permanent);

    let keys = [observed("ed25519:b", b"BBBB", 100)];
    let relayed = merge(
        &db,
        &update("o", KeySighting::Notary, "n.example", &keys, 2),
    )
    .unwrap();
    assert_eq!(relayed[0].tier, KeyTier::Provisional);
    assert_eq!(relayed[0].source, b"n.example");

    let all = read_server_keys(db.server_info(), b"o").unwrap();
    assert_eq!(all.len(), 2);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn the_same_body_refreshes_times_and_keeps_the_source() {
    let (db, root) = open("same");
    let keys = [observed("ed25519:a", b"AAAA", 100)];
    merge(
        &db,
        &update("o", KeySighting::Notary, "n.example", &keys, 1),
    )
    .unwrap();

    let keys = [observed("ed25519:a", b"AAAA", 900)];
    let again = merge(
        &db,
        &update("o", KeySighting::Notary, "m.example", &keys, 5),
    )
    .unwrap();
    assert_eq!(again[0].valid_until_ts, 900);
    assert_eq!(again[0].ts_added_ms, 5);
    assert_eq!(again[0].source, b"n.example", "the original source stays");
    assert_eq!(again[0].tier, KeyTier::Provisional);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// Pins the current choice: a same-body sighting replaces `valid_until_ts`
/// exactly, even with an earlier time. Revisit if the MSC requires monotonic.
#[test]
fn the_same_body_replaces_validity_exactly_even_backwards() {
    let (db, root) = open("validity");
    let keys = [observed("ed25519:a", b"AAAA", 900)];
    merge(&db, &update("o", KeySighting::Direct, "d", &keys, 1)).unwrap();
    let keys = [observed("ed25519:a", b"AAAA", 100)];
    let again = merge(&db, &update("o", KeySighting::Direct, "d", &keys, 2)).unwrap();
    assert_eq!(again[0].valid_until_ts, 100);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn invalid_updates_are_rejected_before_any_write() {
    let (db, root) = open("invalid");
    let good = observed("ed25519:a", b"AAAA", 1);
    for bad in [observed("", b"AAAA", 1), observed("ed25519:b", b"", 1)] {
        let keys = [good.clone(), bad];
        assert!(merge(&db, &update("o", KeySighting::Direct, "d", &keys, 1)).is_err());
    }
    let keys = [good.clone()];
    assert!(merge(&db, &update("", KeySighting::Direct, "d", &keys, 1)).is_err());
    assert_eq!(read_server_keys(db.server_info(), b"o").unwrap().len(), 0);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_direct_fetch_promotes_the_same_provisional_key() {
    let (db, root) = open("promote");
    let keys = [observed("ed25519:a", b"AAAA", 100)];
    merge(
        &db,
        &update("o", KeySighting::Notary, "n.example", &keys, 1),
    )
    .unwrap();
    let confirmed = merge(&db, &update("o", KeySighting::Direct, "origin", &keys, 2)).unwrap();
    assert_eq!(confirmed[0].tier, KeyTier::Permanent);
    assert_eq!(confirmed[0].source, b"n.example");
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_direct_fetch_overrides_only_a_provisional_binding() {
    let (db, root) = open("override");
    let provisional = [observed("ed25519:a", b"AAAA", 100)];
    merge(
        &db,
        &update("o", KeySighting::Notary, "n.example", &provisional, 1),
    )
    .unwrap();

    // A notary cannot replace a provisional binding.
    let other = [observed("ed25519:a", b"XXXX", 200)];
    let kept = merge(
        &db,
        &update("o", KeySighting::Notary, "m.example", &other, 2),
    )
    .unwrap();
    assert_eq!(kept[0].key, b"AAAA");
    assert_eq!(kept[0].ts_added_ms, 1, "an ignored key changes nothing");

    // A direct fetch does, and the new binding is promoted.
    let replaced = merge(&db, &update("o", KeySighting::Direct, "origin", &other, 3)).unwrap();
    assert_eq!(replaced[0].key, b"XXXX");
    assert_eq!(replaced[0].tier, KeyTier::Permanent);
    assert_eq!(replaced[0].source, b"origin");

    // A promoted binding then holds against everything.
    let third = [observed("ed25519:a", b"ZZZZ", 300)];
    for sighting in [KeySighting::Direct, KeySighting::Notary] {
        let held = merge(&db, &update("o", sighting, "x", &third, 4)).unwrap();
        assert_eq!(held[0].key, b"XXXX");
    }
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_repeated_key_id_in_one_batch_sees_the_earlier_result() {
    let (db, root) = open("repeat");
    let keys = [
        observed("ed25519:a", b"AAAA", 100),
        observed("ed25519:a", b"XXXX", 200),
        observed("ed25519:b", b"BBBB", 100),
    ];
    let merged = merge(&db, &update("o", KeySighting::Direct, "origin", &keys, 1)).unwrap();
    assert_eq!(merged.len(), 2, "one result per distinct key id");
    assert_eq!(merged[0].key, b"AAAA", "the first one seen wins");
    assert_eq!(merged[1].key_id, b"ed25519:b");
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn origins_are_isolated_and_reads_are_ordered_and_filterable() {
    let (db, root) = open("isolated");
    let keys = [
        observed("ed25519:b", b"BBBB", 1),
        observed("ed25519:a", b"AAAA", 1),
    ];
    merge(&db, &update("one", KeySighting::Notary, "n", &keys, 1)).unwrap();
    let keys = [observed("ed25519:a", b"CCCC", 1)];
    merge(&db, &update("two", KeySighting::Direct, "d", &keys, 1)).unwrap();

    let one = read_server_keys(db.server_info(), b"one").unwrap();
    let ids: Vec<_> = one.iter().map(|key| key.key_id.clone()).collect();
    assert_eq!(ids, [b"ed25519:a".to_vec(), b"ed25519:b".to_vec()]);
    assert!(one.iter().all(|key| key.tier == KeyTier::Provisional));

    let two = read_server_keys(db.server_info(), b"two").unwrap();
    assert_eq!(two.len(), 1);
    assert_eq!(two[0].key, b"CCCC");
    assert_eq!(
        read_server_keys(db.server_info(), b"three").unwrap().len(),
        0
    );

    let single = read_server_key(db.server_info(), b"one", b"ed25519:a")
        .unwrap()
        .unwrap();
    assert_eq!(single.key, b"AAAA");
    assert!(read_server_key(db.server_info(), b"one", b"ed25519:zzz")
        .unwrap()
        .is_none());
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn the_notary_payload_is_stored_verbatim_with_the_keys() {
    let (db, root) = open("notary");
    assert!(read_notary_payload(db.server_info(), b"o")
        .unwrap()
        .is_none());
    // Spacing and key order that no re-encoder would reproduce.
    let payload = br#"{ "verify_keys":{},  "server_name":"o" , "signatures":{}}"#;
    let keys = [observed("ed25519:a", b"AAAA", 1)];
    let mut batch = update("o", KeySighting::Direct, "origin", &keys, 1);
    batch.notary = Some(payload);
    merge(&db, &batch).unwrap();
    assert_eq!(
        read_notary_payload(db.server_info(), b"o")
            .unwrap()
            .unwrap(),
        payload
    );
    assert!(read_notary_payload(db.server_info(), b"other")
        .unwrap()
        .is_none());

    // A notary-only update publishes no keys.
    let mut only = update("o", KeySighting::Direct, "origin", &[], 2);
    only.notary = Some(b"{}");
    assert_eq!(merge(&db, &only).unwrap().len(), 0);
    assert_eq!(
        read_notary_payload(db.server_info(), b"o")
            .unwrap()
            .unwrap(),
        b"{}"
    );
    assert_eq!(read_server_keys(db.server_info(), b"o").unwrap().len(), 1);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_record_in_the_wrong_slot_is_rejected() {
    let (db, root) = open("wrong-slot");
    let key = ServerKey {
        key_id: b"ed25519:a".to_vec(),
        key: b"AAAA".to_vec(),
        valid_until_ts: 1,
        source: b"s".to_vec(),
        ts_added_ms: 1,
        tier: KeyTier::Permanent,
        expired_ts: None,
        promote_by_ms: 0,
        digest: key_digest(b"AAAA"),
        first_seen_ms: 1,
        last_active_ms: 1,
        corroborated: false,
        inferred_retired: false,
    };
    // A record written under `one`'s collection that names another origin.
    let tx = db.begin_transaction();
    tx.put(
        SERVER_INFO_POOL,
        origin_keys_collection_id(b"one"),
        origin_key_record_id(b"ed25519:a"),
        &NodeData::new(Bytes::from(encode_server_key_record(b"two", &key).unwrap())),
    )
    .unwrap();
    tx.commit().unwrap();
    assert!(read_server_keys(db.server_info(), b"one").is_err());
    assert!(read_server_key(db.server_info(), b"one", b"ed25519:a").is_err());
    let keys = [observed("ed25519:a", b"AAAA", 1)];
    assert!(merge(&db, &update("one", KeySighting::Direct, "d", &keys, 1)).is_err());
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn overlapping_writers_never_lose_a_key() {
    let (db, root) = open("race");
    let db = Arc::new(db);
    let workers: Vec<_> = (0..4_u8)
        .map(|worker| {
            let db = Arc::clone(&db);
            std::thread::spawn(move || {
                for round in 0..10_u8 {
                    let keys = [
                        observed(&format!("ed25519:{worker}-{round}"), &[worker, round], 1),
                        observed("ed25519:shared", b"SHAR", 1),
                    ];
                    merge(
                        &db,
                        &update("o", KeySighting::Direct, "d", &keys, u64::from(round)),
                    )
                    .unwrap();
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    let all = read_server_keys(db.server_info(), b"o").unwrap();
    assert_eq!(
        all.len(),
        4 * 10 + 1,
        "every worker's keys and the shared one"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_direct_fetch_cannot_override_a_closed_expired_or_retired_provisional() {
    let other = [observed("ed25519:a", b"XXXX", 5_000)];
    let cases: [(&str, ObservedKey, u64); 3] = [
        ("window", observed("ed25519:a", b"AAAA", 5_000), WINDOW + 2),
        ("expired", observed("ed25519:a", b"AAAA", 10), 50),
        ("retired", retired("ed25519:a", b"AAAA", 5_000, 20), 50),
    ];
    for (name, first, now) in cases {
        let (db, root) = open(&format!("frozen-{name}"));
        let provisional = [first];
        merge(&db, &update("o", KeySighting::Notary, "n", &provisional, 1)).unwrap();
        let merged =
            update_server_keys(&db, &update("o", KeySighting::Direct, "d", &other, now)).unwrap();
        assert_eq!(merged.keys[0].key, b"AAAA", "{name}: the binding holds");
        assert_eq!(
            merged.issues[0].kind,
            KeyIssueKind::Collision,
            "{name}: reported"
        );
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }
}

#[test]
fn each_provisional_binding_keeps_its_own_window() {
    let (db, root) = open("windows");
    let first = [observed("ed25519:a", b"AAAA", 9_000)];
    merge(&db, &update("o", KeySighting::Notary, "n", &first, 1)).unwrap();
    // A later sighting of another key does not move the first key's deadline.
    let second = [observed("ed25519:b", b"BBBB", 9_000)];
    merge(&db, &update("o", KeySighting::Notary, "n", &second, 900)).unwrap();
    let again = merge(&db, &update("o", KeySighting::Notary, "n", &first, 950)).unwrap();
    assert_eq!(again[0].promote_by_ms, 1 + WINDOW, "a refresh keeps it");
    let a = [observed("ed25519:a", b"XXXX", 9_000)];
    let b = [observed("ed25519:b", b"YYYY", 9_000)];
    let now = 1 + WINDOW + 1;
    assert_eq!(
        merge(&db, &update("o", KeySighting::Direct, "d", &a, now)).unwrap()[0].key,
        b"AAAA"
    );
    assert_eq!(
        merge(&db, &update("o", KeySighting::Direct, "d", &b, now)).unwrap()[0].key,
        b"YYYY"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn the_first_expired_ts_wins_and_a_future_one_is_dropped() {
    let (db, root) = open("expired");
    let active = [observed("ed25519:a", b"AAAA", 9_000)];
    merge(&db, &update("o", KeySighting::Direct, "d", &active, 1_000)).unwrap();

    // A future value is malformed for that entry and consumes nothing.
    let future = [retired(
        "ed25519:a",
        b"AAAA",
        9_000,
        1_000 + EXPIRED_TS_SKEW_MS + 1,
    )];
    let merged =
        update_server_keys(&db, &update("o", KeySighting::Direct, "d", &future, 1_000)).unwrap();
    assert!(merged.keys.is_empty(), "the entry is dropped");
    assert_eq!(merged.issues[0].kind, KeyIssueKind::FutureExpiredTs);
    assert_eq!(
        read_server_key(db.server_info(), b"o", b"ed25519:a")
            .unwrap()
            .unwrap()
            .expired_ts,
        None
    );

    // The first sane value is assigned, prospectively and without comment.
    let first = [retired("ed25519:a", b"AAAA", 9_000, 900)];
    let merged =
        update_server_keys(&db, &update("o", KeySighting::Direct, "d", &first, 1_000)).unwrap();
    assert_eq!(merged.keys[0].expired_ts, Some(900));
    assert_eq!(merged.issues.len(), 0);

    // A different one, earlier or later, is reported and ignored.
    for other in [800, 950] {
        let later = [retired("ed25519:a", b"AAAA", 9_000, other)];
        let merged =
            update_server_keys(&db, &update("o", KeySighting::Direct, "d", &later, 1_000)).unwrap();
        assert_eq!(merged.keys[0].expired_ts, Some(900));
        assert_eq!(merged.issues[0].kind, KeyIssueKind::ExpiredTsConflict);
    }
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_rejected_collision_leaves_the_served_payload_alone() {
    let (db, root) = open("served");
    let keys = [observed("ed25519:a", b"AAAA", 9_000)];
    let mut first = update("o", KeySighting::Direct, "d", &keys, 1);
    first.notary = Some(b"first");
    merge(&db, &first).unwrap();

    let clash = [observed("ed25519:a", b"XXXX", 9_000)];
    let mut second = update("o", KeySighting::Direct, "d", &clash, 2);
    second.notary = Some(b"second");
    let merged = update_server_keys(&db, &second).unwrap();
    assert_eq!(merged.issues[0].kind, KeyIssueKind::Collision);
    assert_eq!(
        read_notary_payload(db.server_info(), b"o")
            .unwrap()
            .unwrap(),
        b"first"
    );

    // A displacing direct fetch is accepted, so its payload is served.
    let (db2, root2) = open("served-displace");
    let mut provisional = update("o", KeySighting::Notary, "n", &keys, 1);
    provisional.notary = Some(b"first");
    merge(&db2, &provisional).unwrap();
    let mut direct = update("o", KeySighting::Direct, "d", &clash, 2);
    direct.notary = Some(b"second");
    let merged = update_server_keys(&db2, &direct).unwrap();
    assert_eq!(merged.issues[0].kind, KeyIssueKind::Displaced);
    assert_eq!(
        read_notary_payload(db2.server_info(), b"o")
            .unwrap()
            .unwrap(),
        b"second"
    );
    drop((db, db2));
    let _ = std::fs::remove_dir_all(root);
    let _ = std::fs::remove_dir_all(root2);
}

fn with_limits(mut batch: KeyUpdate<'_>, limits: KeyLimits) -> KeyUpdate<'_> {
    batch.limits = limits;
    batch
}

fn snapshot(mut batch: KeyUpdate<'_>) -> KeyUpdate<'_> {
    batch.snapshot = true;
    batch
}

#[test]
fn a_direct_snapshot_retires_the_keys_it_leaves_out() {
    let (db, root) = open("inferred");
    let both = [
        observed("ed25519:a", b"AAAA", 9_000),
        observed("ed25519:b", b"BBBB", 9_000),
    ];
    merge(&db, &update("o", KeySighting::Direct, "d", &both, 100)).unwrap();

    // A notary snapshot that omits `a` retires nothing: only the origin's
    // own responses do.
    let only_b = [observed("ed25519:b", b"BBBB", 9_000)];
    merge(
        &db,
        &snapshot(update("o", KeySighting::Notary, "n", &only_b, 200)),
    )
    .unwrap();
    let a = read_server_key(db.server_info(), b"o", b"ed25519:a")
        .unwrap()
        .unwrap();
    assert_eq!(a.retirement_ts(), None);

    merge(
        &db,
        &snapshot(update("o", KeySighting::Direct, "d", &only_b, 300)),
    )
    .unwrap();
    let a = read_server_key(db.server_info(), b"o", b"ed25519:a")
        .unwrap()
        .unwrap();
    assert_eq!(
        a.retirement_ts(),
        Some(100),
        "retired when last seen active"
    );
    assert!(a.corroborated);

    // Seeing it active again undoes the inference.
    merge(&db, &update("o", KeySighting::Direct, "d", &both, 400)).unwrap();
    let a = read_server_key(db.server_info(), b"o", b"ed25519:a")
        .unwrap()
        .unwrap();
    assert_eq!(a.retirement_ts(), None);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn retired_keys_are_evicted_in_the_msc_order_and_keep_their_digest() {
    let (db, root) = open("evict");
    let limits = KeyLimits {
        retired_retained: 2,
        ..KeyLimits::default()
    };
    // Active once, so corroborated, then retired at 100.
    let active = [observed("ed25519:c", b"CCCC", 9_000)];
    merge(&db, &update("o", KeySighting::Direct, "d", &active, 50)).unwrap();
    let old = [
        retired("ed25519:c", b"CCCC", 9_000, 100),
        // Never seen active: uncorroborated, though retired later.
        retired("ed25519:u", b"UUUU", 9_000, 900),
        retired("ed25519:z", b"ZZZZ", 9_000, 500),
        retired("ed25519:y", b"YYYY", 9_000, 500),
    ];
    let merged = update_server_keys(
        &db,
        &with_limits(update("o", KeySighting::Direct, "d", &old, 1_000), limits),
    )
    .unwrap();
    // Corroborated `c` first; then the later retirement `u`; `y` and `z` tie
    // on time and sort by key id, but only one slot is left after `c` and `u`.
    let kept: Vec<_> = read_server_keys(db.server_info(), b"o")
        .unwrap()
        .into_iter()
        .filter(|key| !key.is_evicted())
        .map(|key| key.key_id)
        .collect();
    assert_eq!(kept, [b"ed25519:c".to_vec(), b"ed25519:u".to_vec()]);
    let kinds: Vec<_> = merged.issues.iter().map(|issue| issue.kind).collect();
    assert_eq!(kinds, [KeyIssueKind::Evicted; 2]);

    // The evicted key id still collides with a different body.
    let clash = [retired("ed25519:y", b"XXXX", 9_000, 500)];
    let again = update_server_keys(
        &db,
        &with_limits(update("o", KeySighting::Direct, "d", &clash, 1_100), limits),
    )
    .unwrap();
    assert_eq!(again.issues[0].kind, KeyIssueKind::Collision);
    assert!(again.keys[0].is_evicted());
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn evicting_a_corroborated_key_is_flagged() {
    let (db, root) = open("flood");
    let limits = KeyLimits {
        retired_retained: 1,
        ..KeyLimits::default()
    };
    let active = [
        observed("ed25519:a", b"AAAA", 9_000),
        observed("ed25519:b", b"BBBB", 9_000),
    ];
    merge(&db, &update("o", KeySighting::Direct, "d", &active, 10)).unwrap();
    let old = [
        retired("ed25519:a", b"AAAA", 9_000, 20),
        retired("ed25519:b", b"BBBB", 9_000, 30),
    ];
    let merged = update_server_keys(
        &db,
        &with_limits(update("o", KeySighting::Direct, "d", &old, 40), limits),
    )
    .unwrap();
    assert_eq!(merged.issues.len(), 1);
    assert_eq!(merged.issues[0].key_id, b"ed25519:a");
    assert_eq!(merged.issues[0].kind, KeyIssueKind::EvictedCorroborated);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn an_evicted_key_that_returns_competes_for_a_slot_again() {
    let (db, root) = open("return");
    let limits = KeyLimits {
        retired_retained: 1,
        ..KeyLimits::default()
    };
    let old = [
        retired("ed25519:a", b"AAAA", 9_000, 10),
        retired("ed25519:b", b"BBBB", 9_000, 20),
    ];
    merge(
        &db,
        &with_limits(update("o", KeySighting::Direct, "d", &old, 30), limits),
    )
    .unwrap();
    let again = [retired("ed25519:a", b"AAAA", 9_000, 10)];
    let merged = update_server_keys(
        &db,
        &with_limits(update("o", KeySighting::Direct, "d", &again, 40), limits),
    )
    .unwrap();
    assert!(merged.keys[0].is_evicted(), "it still sorts below `b`");
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn the_digest_cap_refuses_new_key_ids_per_bucket_and_promotion_moves_one() {
    let (db, root) = open("cap");
    let limits = KeyLimits {
        digest_cap: 1,
        ..KeyLimits::default()
    };
    let capped = |sighting, keys, now| with_limits(update("o", sighting, "s", keys, now), limits);

    let a = [observed("ed25519:a", b"AAAA", 9_000)];
    merge(&db, &capped(KeySighting::Notary, &a, 1)).unwrap();
    let b = [observed("ed25519:b", b"BBBB", 9_000)];
    let refused = update_server_keys(&db, &capped(KeySighting::Notary, &b, 2)).unwrap();
    assert_eq!(refused.issues[0].kind, KeyIssueKind::DigestCapReached);
    assert!(refused.keys.is_empty(), "no digest binding, no key");

    // The direct bucket is separate, so a notary flood does not block it.
    let direct = merge(&db, &capped(KeySighting::Direct, &b, 3)).unwrap();
    assert_eq!(direct[0].key, b"BBBB");

    // Promotion moves `a` to the direct bucket even though that is full.
    let promoted = merge(&db, &capped(KeySighting::Direct, &a, 4)).unwrap();
    assert_eq!(promoted[0].tier, KeyTier::Permanent);

    // And it freed the notary bucket.
    let c = [observed("ed25519:c", b"CCCC", 9_000)];
    let fresh = update_server_keys(&db, &capped(KeySighting::Notary, &c, 5)).unwrap();
    assert_eq!(fresh.issues.len(), 0);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_response_over_the_ceilings_or_with_a_long_key_id_is_rejected() {
    let (db, root) = open("ceilings");
    let limits = KeyLimits {
        active_per_response: 1,
        retired_per_response: 1,
        ..KeyLimits::default()
    };
    let two_active = [
        observed("ed25519:a", b"AAAA", 1),
        observed("ed25519:b", b"BBBB", 1),
    ];
    let two_retired = [
        retired("ed25519:a", b"AAAA", 1, 1),
        retired("ed25519:b", b"BBBB", 1, 1),
    ];
    for keys in [&two_active[..], &two_retired[..]] {
        assert!(update_server_keys(
            &db,
            &with_limits(update("o", KeySighting::Direct, "d", keys, 1), limits)
        )
        .is_err());
    }
    let long = [observed(&"k".repeat(256), b"AAAA", 1)];
    assert!(update_server_keys(&db, &update("o", KeySighting::Direct, "d", &long, 1)).is_err());
    let edge = [observed(&"k".repeat(255), b"AAAA", 1)];
    assert!(update_server_keys(&db, &update("o", KeySighting::Direct, "d", &edge, 1)).is_ok());
    assert_eq!(read_server_keys(db.server_info(), b"o").unwrap().len(), 1);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn concurrent_inserts_cannot_overrun_the_digest_cap() {
    let (db, root) = open("cap-race");
    let db = Arc::new(db);
    let limits = KeyLimits {
        digest_cap: 5,
        ..KeyLimits::default()
    };
    let workers: Vec<_> = (0..4_u8)
        .map(|worker| {
            let db = Arc::clone(&db);
            std::thread::spawn(move || {
                for round in 0..10_u8 {
                    let keys = [observed(
                        &format!("ed25519:{worker}-{round}"),
                        &[worker, round],
                        1,
                    )];
                    let batch =
                        with_limits(update("o", KeySighting::Direct, "d", &keys, 1), limits);
                    update_server_keys(&db, &batch).unwrap();
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    let all = read_server_keys(db.server_info(), b"o").unwrap();
    assert_eq!(all.len(), 5, "the cap holds exactly under racing inserts");
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}
