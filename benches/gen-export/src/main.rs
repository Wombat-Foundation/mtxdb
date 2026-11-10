//! Generate a valid, signed `mtxdb-export/v1` room and its keys file.
//!
//! Events are built, content-hashed, signed and given reference-hash ids with
//! the same `rezzy` code the importer verifies with, so an import of the output
//! exercises the real findings path instead of the invalid-signature path.
//! Output is a pure function of the arguments.
//!
//! ```text
//! gen-export --mode room  --events 24000 --users 2000 --out /tmp/gen/room1x
//! gen-export --mode chain --events 4000  --out /tmp/gen/chain
//! ```
//!
//! `room` mirrors the large real rooms: nearly every event is an
//! `m.room.member` state event, mostly profile updates by users who already
//! joined, with a few topic/name events, forks and up to three `prev_events`.
//! `--message-per-mille` mixes in `m.room.message` timeline events, which is
//! the shape of the message-heavy real rooms (about 20% state).
//! `chain` emits one `m.room.power_levels` update after another, each naming
//! the last as an auth event, so auth-closure size grows with the chain.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD};
use base64::Engine as _;
use ed25519_zebra::{SigningKey, VerificationKey};
use rezzy::json::Value;

const ROOM_VERSION: &str = "10";
const SERVER: &str = "gen.test";
const KEY_ID: &str = "ed25519:gen1";
const BASE_TS: u64 = 1_700_000_000_000;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn chance(&mut self, per_mille: u64) -> bool {
        self.next() % 1000 < per_mille
    }
}

fn s(text: &str) -> Value {
    Value::String(text.to_owned())
}

fn obj(fields: Vec<(&str, Value)>) -> Value {
    let mut map = BTreeMap::new();
    for (key, value) in fields {
        map.insert(key.to_owned(), value);
    }
    Value::Object(map.into_iter().collect())
}

fn ids(list: &[String]) -> Value {
    Value::Array(list.iter().map(|id| s(id)).collect())
}

struct Gen {
    key: SigningKey,
    room_id: String,
    ts: u64,
    out: String,
    count: u64,
}

struct Made {
    id: String,
    depth: u64,
}

impl Gen {
    #[allow(clippy::too_many_arguments)]
    fn event(
        &mut self,
        kind: &str,
        state_key: Option<&str>,
        sender: &str,
        content: Value,
        prev: &[(String, u64)],
        auth: &[String],
    ) -> Made {
        let depth = prev.iter().map(|(_, d)| *d).max().unwrap_or(0) + 1;
        let prev_ids: Vec<String> = prev.iter().map(|(id, _)| id.clone()).collect();
        self.ts += 1000;
        let mut fields = vec![
            ("auth_events", ids(auth)),
            ("content", content),
            ("depth", Value::from(depth)),
            ("origin", s(SERVER)),
            ("origin_server_ts", Value::from(self.ts)),
            ("prev_events", ids(&prev_ids)),
            ("room_id", s(&self.room_id)),
            ("sender", s(sender)),
            ("type", s(kind)),
        ];
        if let Some(key) = state_key {
            fields.push(("state_key", s(key)));
        }
        let mut pdu = obj(fields);
        let hash = rezzy::compute_content_hash(&pdu, ROOM_VERSION).expect("content hash");
        let Value::Object(map) = &mut pdu else {
            unreachable!()
        };
        map.insert("hashes".to_owned(), obj(vec![("sha256", s(&hash))]));
        let message = rezzy::try_canonical_redacted_json(&pdu, ROOM_VERSION).expect("canonical");
        let signature = STANDARD_NO_PAD.encode(<[u8; 64]>::from(self.key.sign(message.as_bytes())));
        let Value::Object(map) = &mut pdu else {
            unreachable!()
        };
        map.insert(
            "signatures".to_owned(),
            obj(vec![(SERVER, obj(vec![(KEY_ID, s(&signature))]))]),
        );
        let id = format!(
            "${}",
            rezzy::reference_hash(&pdu, ROOM_VERSION).expect("reference hash")
        );
        let line = obj(vec![
            ("auth_events", ids(auth)),
            ("depth", Value::from(depth)),
            ("event_id", s(&id)),
            ("outlier", Value::Bool(false)),
            ("pdu", pdu),
            ("prev_events", ids(&prev_ids)),
            ("t", s("event")),
        ]);
        writeln!(self.out, "{line}").unwrap();
        self.count += 1;
        Made { id, depth }
    }
}

fn user(n: usize) -> String {
    format!("@u{n}:{SERVER}")
}

fn main() {
    let mut mode = "room".to_owned();
    let mut events = 24_000usize;
    let mut users = 2_000usize;
    let mut seed = 1u64;
    let mut fork_per_mille = 40u64;
    let mut state_per_mille = 1u64;
    let mut message_per_mille = 0u64;
    let mut out_dir = PathBuf::from("gen-out");
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().expect("every flag takes a value");
        match flag.as_str() {
            "--mode" => mode = value,
            "--events" => events = value.parse().expect("--events"),
            "--users" => users = value.parse().expect("--users"),
            "--seed" => seed = value.parse().expect("--seed"),
            "--fork-per-mille" => fork_per_mille = value.parse().expect("--fork-per-mille"),
            "--state-per-mille" => state_per_mille = value.parse().expect("--state-per-mille"),
            "--message-per-mille" => {
                message_per_mille = value.parse().expect("--message-per-mille");
            }
            "--out" => out_dir = PathBuf::from(value),
            other => panic!("unknown flag {other}"),
        }
    }

    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut secret = [0u8; 32];
    for chunk in secret.chunks_mut(8) {
        chunk.copy_from_slice(&rng.next().to_le_bytes());
    }
    let key = SigningKey::from(secret);
    let public = <[u8; 32]>::from(VerificationKey::from(&key));
    let room_id = format!("!gen{seed}x{events}:{SERVER}");
    let mut gen = Gen {
        key,
        room_id: room_id.clone(),
        ts: BASE_TS,
        out: String::new(),
        count: 0,
    };

    let creator = user(0);
    let create = gen.event(
        "m.room.create",
        Some(""),
        &creator,
        obj(vec![
            ("creator", s(&creator)),
            ("room_version", s(ROOM_VERSION)),
        ]),
        &[],
        &[],
    );
    let mut tips = vec![(create.id.clone(), create.depth)];
    let push = |tips: &mut Vec<(String, u64)>, made: &Made| {
        tips.clear();
        tips.push((made.id.clone(), made.depth));
    };
    let creator_join = gen.event(
        "m.room.member",
        Some(&creator),
        &creator,
        obj(vec![("membership", s("join"))]),
        &tips,
        std::slice::from_ref(&create.id),
    );
    push(&mut tips, &creator_join);
    let levels = |extra: &[usize]| {
        let mut users_map = BTreeMap::new();
        users_map.insert(creator.clone(), Value::from(100u64));
        for &n in extra {
            users_map.insert(user(n), Value::from(0u64));
        }
        obj(vec![
            ("ban", Value::from(50u64)),
            ("events_default", Value::from(0u64)),
            ("invite", Value::from(0u64)),
            ("kick", Value::from(50u64)),
            ("redact", Value::from(50u64)),
            ("state_default", Value::from(50u64)),
            ("users", Value::Object(users_map.into_iter().collect())),
            ("users_default", Value::from(0u64)),
        ])
    };
    let mut power = gen.event(
        "m.room.power_levels",
        Some(""),
        &creator,
        levels(&[]),
        &tips,
        &[create.id.clone(), creator_join.id.clone()],
    );
    push(&mut tips, &power);
    let join_rules = gen.event(
        "m.room.join_rules",
        Some(""),
        &creator,
        obj(vec![("join_rule", s("public"))]),
        &tips,
        &[create.id.clone(), power.id.clone(), creator_join.id.clone()],
    );
    push(&mut tips, &join_rules);

    let mut member: Vec<Option<(String, u64)>> = vec![None; users.max(1)];
    member[0] = Some((creator_join.id.clone(), 0));
    let mut joined = 1usize;
    let mut chain_extra: Vec<usize> = Vec::new();
    let mut revision = 0u64;
    while (gen.count as usize) < events {
        revision += 1;
        let prev_tips = tips.clone();
        if mode == "chain" {
            chain_extra.push(chain_extra.len() + 1);
            let made = gen.event(
                "m.room.power_levels",
                Some(""),
                &creator,
                levels(&chain_extra),
                &prev_tips,
                &[create.id.clone(), power.id.clone(), creator_join.id.clone()],
            );
            power = made;
            push(&mut tips, &power);
            continue;
        }
        let made;
        if rng.chance(state_per_mille) {
            let kind = if rng.chance(500) {
                "m.room.topic"
            } else {
                "m.room.name"
            };
            let field = if kind == "m.room.topic" {
                "topic"
            } else {
                "name"
            };
            made = gen.event(
                kind,
                Some(""),
                &creator,
                obj(vec![(field, s(&format!("{field} {revision}")))]),
                &prev_tips,
                &[create.id.clone(), power.id.clone(), creator_join.id.clone()],
            );
        } else if joined >= 2 && rng.chance(message_per_mille) {
            // A timeline event: no state key, so it leaves the state as it was.
            let who = rng.below(joined);
            let sender = user(who);
            let previous = member[who].clone().expect("joined user has a member event");
            made = gen.event(
                "m.room.message",
                None,
                &sender,
                obj(vec![
                    ("body", s(&format!("message {revision}"))),
                    ("msgtype", s("m.text")),
                ]),
                &prev_tips,
                &[create.id.clone(), power.id.clone(), previous.0],
            );
        } else if joined < users && (joined < 2 || rng.chance(150)) {
            let who = joined;
            joined += 1;
            let sender = user(who);
            made = gen.event(
                "m.room.member",
                Some(&sender),
                &sender,
                obj(vec![
                    ("displayname", s(&format!("user {who}"))),
                    ("membership", s("join")),
                ]),
                &prev_tips,
                &[create.id.clone(), power.id.clone(), join_rules.id.clone()],
            );
            member[who] = Some((made.id.clone(), made.depth));
        } else {
            let who = rng.below(joined);
            let sender = user(who);
            let previous = member[who].clone().expect("joined user has a member event");
            made = gen.event(
                "m.room.member",
                Some(&sender),
                &sender,
                obj(vec![
                    ("displayname", s(&format!("user {who} v{revision}"))),
                    ("membership", s("join")),
                ]),
                &prev_tips,
                &[
                    create.id.clone(),
                    power.id.clone(),
                    join_rules.id.clone(),
                    previous.0,
                ],
            );
            member[who] = Some((made.id.clone(), made.depth));
        }
        // Usually the new event is the only tip. A fork leaves the old tips
        // and the new event as tips, so the next event merges them (at most
        // three).
        if rng.chance(fork_per_mille) && tips.len() < 3 {
            tips.push((made.id.clone(), made.depth));
        } else {
            push(&mut tips, &made);
        }
    }

    fs::create_dir_all(&out_dir).expect("create output dir");
    let header = obj(vec![
        ("format", s("mtxdb-export/v1")),
        ("room_id", s(&room_id)),
        ("server", s(SERVER)),
        ("t", s("header")),
        ("timeline", Value::Bool(true)),
    ]);
    let mut text = format!("{header}\n");
    text.push_str(&gen.out);
    let end = obj(vec![("events", Value::from(gen.count)), ("t", s("end"))]);
    writeln!(text, "{end}").unwrap();
    fs::write(out_dir.join("room-gen.jsonl"), text).expect("write room");
    let keys = obj(vec![
        ("key", s(&STANDARD.encode(public))),
        ("key_id", s(KEY_ID)),
        ("kind", s("permanent")),
        ("origin", s(SERVER)),
        ("source", s(SERVER)),
        ("t", s("server_keys")),
        ("ts_added_ms", Value::from(BASE_TS)),
        ("valid_until_ts", Value::from(4_102_444_800_000u64)),
    ]);
    fs::write(out_dir.join("server-keys.jsonl"), format!("{keys}\n")).expect("write keys");
    eprintln!(
        "wrote {} events ({mode}, users {joined}) to {}",
        gen.count,
        out_dir.display()
    );
}
