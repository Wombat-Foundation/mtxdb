//! Matrix room state HAMTs: the shared implementation lives in
//! [`mtxdb::state_hamt`] (feature `state-hamt`), so the importer and server
//! integrations write the same bytes. This module re-exports it for the CLI and
//! keeps the CLI's golden tests.

#[cfg(test)]
use std::collections::HashSet;
#[cfg(test)]
use std::sync::Arc;
#[cfg(test)]
use std::time::Instant;

#[cfg(test)]
use rezzy::hamt::{HamtNode, NodeRef, PersistedInternalNode};
#[cfg(test)]
use rezzy::hamt_root::StateHamtKey;
#[cfg(test)]
use rezzy::incremental::LtHash;

pub(crate) use mtxdb::state_hamt::*;
#[cfg(test)]
use mtxdb::DigestAlgorithm;

/// Base64url-encode a state-group `LtHash` digest, for golden tests.
#[cfg(test)]
#[must_use]
pub(crate) fn encode_state_group_id(id: &[u8; 32]) -> String {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine as _;
    URL_SAFE_NO_PAD.encode(id)
}

/// The state-group id (unkeyed `LtHash` digest) for a set of logical entries.
#[cfg(test)]
#[must_use]
pub(crate) fn state_group_id(entries: &[(String, String, String)]) -> [u8; 32] {
    let mut lattice = LtHash::default();
    for (event_type, state_key, event_id) in entries {
        lattice.insert(event_type, state_key, event_id);
    }
    rezzy::hamt::state_group_id_from_lthash(&lattice)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries() -> Vec<(String, String, String)> {
        vec![
            (
                "m.room.create".to_owned(),
                String::new(),
                "$create:example.org".to_owned(),
            ),
            (
                "m.room.member".to_owned(),
                "@alice:example.org".to_owned(),
                "$member:example.org".to_owned(),
            ),
            (
                "m.room.name".to_owned(),
                String::new(),
                "$name:example.org".to_owned(),
            ),
        ]
    }

    #[test]
    fn build_is_deterministic_and_uses_lthash_identity() {
        let input = entries();
        let first = build_state_hamt("!room:example.org", b"prefix", &input).unwrap();
        let second = build_state_hamt("!room:example.org", b"prefix", &input).unwrap();

        assert_eq!(first.state_group_id, state_group_id(&input));
        assert_eq!(first.root_record, second.root_record);
        assert_eq!(first.nodes, second.nodes);
        assert!(first.root_record.starts_with(b"MTHR\x01"));
        assert_ne!(first.nodes, Vec::<([u8; 32], Vec<u8>)>::new());
        assert!(first
            .nodes
            .iter()
            .all(|(_, payload)| payload.starts_with(b"MTHN\x01")));
    }

    /// Repack must lay a state group's records out dependencies-first:
    /// HAMT nodes (deepest first), then the root record, then the group, with
    /// a child group after its parent.
    #[test]
    fn repack_edges_order_state_records_dependencies_first() {
        use mtxdb::csr::Csr;
        use mtxdb::{
            encode_state_group_record, state_hamt_node_id, StateGroupDerivation,
            StateGroupInstance, StateGroupResolver, StateGroupTrust,
        };

        let room = "!room:example.org";
        let big: Vec<(String, String, String)> = (0..400)
            .map(|i| {
                (
                    "m.room.member".to_owned(),
                    format!("@u{i}:x"),
                    format!("$e{i}"),
                )
            })
            .collect();
        let built = build_state_hamt(room, &room_hamt_prefix(room), &big).unwrap();
        assert!(built.nodes.len() > 1, "fixture must nest");

        let root_id = state_hamt_root_node_id("ns", &built.state_group_id);
        let derivation =
            StateGroupDerivation::new(StateGroupResolver::V2, StateGroupTrust::TrustedImport);
        let parent_group = StateGroupInstance {
            parents: Vec::new(),
            lthash: built.state_group_id,
            root_id,
            derivation,
        };
        let parent_id = [0xA1; 16];
        let child_group = StateGroupInstance {
            parents: vec![parent_id],
            lthash: built.state_group_id,
            root_id,
            derivation,
        };
        let child_id = [0xA2; 16];

        let mut records: Vec<([u8; 16], Vec<u8>)> = built
            .nodes
            .iter()
            .map(|(hash, payload)| (state_hamt_node_id(hash), payload.clone()))
            .collect();
        records.push((root_id, built.root_record.clone()));
        records.push((child_id, encode_state_group_record(&child_group)));
        records.push((parent_id, encode_state_group_record(&parent_group)));
        records.reverse(); // dependents first, the worst case for a naive order

        let ids: Vec<[u8; 16]> = records.iter().map(|(id, _)| *id).collect();
        let adjacency: HashMap<[u8; 16], Vec<[u8; 16]>> = records
            .iter()
            .map(|(id, data)| (*id, extract_state_edges(data).expect("state record")))
            .collect();
        let csr = Csr::build_from_edges(&ids, &adjacency);
        let order: Vec<[u8; 16]> = csr
            .layered_order()
            .expect("acyclic")
            .into_iter()
            .map(|local| *csr.hash_of(local).unwrap())
            .collect();
        let position = |id: &[u8; 16]| order.iter().position(|x| x == id).unwrap();

        for (id, targets) in &adjacency {
            for target in targets.iter().filter(|t| ids.contains(t)) {
                assert!(
                    position(target) < position(id),
                    "dependency must come first"
                );
            }
        }
        assert!(position(&root_id) > position(&state_hamt_node_id(&built.root_hash)));
        assert!(position(&parent_id) > position(&root_id));
        assert_eq!(order.last(), Some(&child_id));
    }

    #[test]
    fn state_group_identity_is_order_independent() {
        let mut reversed = entries();
        reversed.reverse();
        assert_eq!(state_group_id(&entries()), state_group_id(&reversed));
        assert_eq!(
            build_state_hamt("!room:example.org", b"prefix", &entries())
                .unwrap()
                .root_record,
            build_state_hamt("!room:example.org", b"prefix", &reversed)
                .unwrap()
                .root_record
        );
    }

    #[test]
    fn leaf_keys_preserve_type_and_state_key() {
        assert_eq!(
            state_hamt_leaf_key("m.room\"name", "line\nkey"),
            ("m.room\"name".into(), "line\nkey".to_owned())
        );
        assert_eq!(room_hamt_prefix("!room:example.org").len(), 8);
    }

    #[test]
    fn msc4291_room_prefix_uses_decoded_hash() {
        let hash = [0x5a; 32];
        let room_id = format!("!{}", encode_base64(&hash));
        assert_eq!(room_hamt_prefix(&room_id), hash[..8]);
    }

    fn encode_base64(bytes: &[u8]) -> String {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine as _;
        URL_SAFE_NO_PAD.encode(bytes)
    }

    use std::collections::HashMap;

    type Node = Arc<HamtNode<StateHamtKey, String>>;
    type Tuple = (String, String, String, u8);
    type Store = HashMap<[u8; 32], Vec<u8>>;

    const STGP_PAYLOAD: usize = 73;
    // One explicit directory entry: 16-byte checkpoint event id + u32 ordinal.
    const DIRECTORY_ENTRY: usize = 20;
    const ROOM: &str = "!space:example.org";
    /// Explicit sweep parameters. The production
    /// `STATE_LATTICE_KEYFRAME_HOPS` configures keyframe spacing in the
    /// writer; it must not drive the experiment, or changing the writer
    /// silently changes what was measured.
    const KEYFRAME_SWEEP: [usize; 3] = [16, 64, 256];

    fn add(left: usize, right: usize) -> usize {
        left.checked_add(right).expect("sum fits")
    }

    fn sub(left: usize, right: usize) -> usize {
        left.checked_sub(right).expect("difference fits")
    }

    fn mul(left: usize, right: usize) -> usize {
        left.checked_mul(right).expect("product fits")
    }

    fn div(left: usize, right: usize) -> usize {
        left.checked_div(right).expect("divisor is nonzero")
    }

    /// The recorded step after `count` applied tuples (`count` is 1-based).
    fn step_at<T>(steps: &[T], count: usize) -> &T {
        &steps[sub(count, 1)]
    }

    /// The `m.space.child` state key: a full child room id. By default a
    /// 43-character hash-style localpart plus a server name (about 56 B);
    /// `MEASURE_KEY=short` restores the earlier 22 B `!child{n}:example.org`.
    fn child_state_key(ordinal: usize) -> String {
        if std::env::var("MEASURE_KEY").is_ok_and(|style| style == "short") {
            return format!("!child{ordinal}:example.org");
        }
        let bytes = DigestAlgorithm::Sha256.digest(&add(ordinal, 1_000_000).to_be_bytes());
        format!("!{}:example.org", encode_base64(&bytes))
    }

    fn event_id(ordinal: usize) -> String {
        let bytes = DigestAlgorithm::Sha256.digest(&ordinal.to_be_bytes());
        format!("${}", encode_base64(&bytes))
    }

    /// Bytes per event as fixed-point tenths, avoiding float casts.
    fn tenths(bytes: usize, over: usize) -> String {
        let value = div(mul(bytes, 10), over.max(1));
        format!("{}.{}", div(value, 10), value.checked_rem(10).unwrap())
    }

    fn hundredths(bytes: usize, over: usize) -> String {
        let value = div(mul(bytes, 100), over.max(1));
        format!("{}.{:02}", div(value, 100), value.checked_rem(100).unwrap())
    }

    /// Exact replay delta: type, state key and event id, each with a
    /// big-endian u16 length prefix. The separately persisted `Verdict`
    /// record gates whether the decoded tuple is applied.
    fn encode_delta(tuple: &Tuple) -> Vec<u8> {
        let mut out = Vec::new();
        for part in [&tuple.0, &tuple.1, &tuple.2] {
            out.extend_from_slice(&u16::try_from(part.len()).unwrap().to_be_bytes());
            out.extend_from_slice(part.as_bytes());
        }
        out
    }

    fn decode_delta(bytes: &[u8]) -> Tuple {
        let mut at = 0;
        let mut parts = Vec::new();
        for _ in 0..3 {
            let len = usize::from(u16::from_be_bytes([bytes[at], bytes[add(at, 1)]]));
            let start = add(at, 2);
            parts.push(String::from_utf8(bytes[start..add(start, len)].to_vec()).unwrap());
            at = add(start, len);
        }
        assert_eq!(at, bytes.len());
        let event = parts.pop().unwrap();
        let state_key = parts.pop().unwrap();
        let kind = parts.pop().unwrap();
        (kind, state_key, event, 0)
    }

    fn persisted_verdict(tuple: &Tuple) -> Vec<u8> {
        let verdict = match tuple.3 {
            2 => mtxdb::Verdict {
                rejection: Some(1),
                ..mtxdb::Verdict::default()
            },
            1 => mtxdb::Verdict {
                soft_fail: Some(1),
                ..mtxdb::Verdict::default()
            },
            _ => mtxdb::Verdict::default(),
        };
        mtxdb::encode_verdict(tuple.2.as_bytes(), &verdict).unwrap()
    }

    /// Reconstruct tuples by decoding the stored delta and the real persisted
    /// verdict record. Rejected wins if both verdict bits are present.
    fn decode_persisted(tuples: &[Tuple]) -> Vec<Tuple> {
        tuples
            .iter()
            .map(|tuple| {
                let mut decoded = decode_delta(&encode_delta(tuple));
                let encoded_verdict = persisted_verdict(tuple);
                let (event_id, verdict) = mtxdb::decode_verdict(&encoded_verdict).unwrap();
                assert_eq!(event_id, decoded.2.as_bytes());
                decoded.3 = if verdict.rejection.is_some() {
                    2
                } else {
                    u8::from(verdict.soft_fail.is_some())
                };
                decoded
            })
            .collect()
    }

    fn mutations(tuples: &[Tuple]) -> Vec<(StateHamtKey, Option<String>)> {
        tuples
            .iter()
            .filter(|tuple| tuple.3 != 2)
            .map(|(kind, state_key, event, _)| {
                (state_hamt_leaf_key(kind, state_key), Some(event.clone()))
            })
            .collect()
    }

    /// Cold/hot node source over stored bytes; counts decodes.
    struct Source<'a> {
        bytes: &'a Store,
        cache: HashMap<[u8; 32], Node>,
        reads: usize,
    }

    impl<'a> Source<'a> {
        fn new(bytes: &'a Store) -> Self {
            Self {
                bytes,
                cache: HashMap::new(),
                reads: 0,
            }
        }

        fn get(&mut self, hash: &[u8; 32]) -> Node {
            if let Some(node) = self.cache.get(hash) {
                return Arc::clone(node);
            }
            self.reads = add(self.reads, 1);
            let bytes = self.bytes.get(hash).expect("node stored");
            let node: Node = Arc::new(
                PersistedInternalNode::<StateHamtKey, String>::decode_v1_verified(
                    bytes,
                    ROOM.as_bytes(),
                    *hash,
                )
                .unwrap(),
            );
            self.cache.insert(*hash, Arc::clone(&node));
            node
        }
    }

    /// Count entries and depth under `node`, resolving lazy children.
    fn walk(node: &Node, source: &mut Source<'_>, depth: usize, totals: &mut (usize, usize)) {
        totals.0 = add(totals.0, node.leaves.len());
        totals.1 = totals.1.max(depth);
        for child in &node.children {
            let next = match child {
                NodeRef::Resolved(resolved) => Arc::clone(resolved),
                NodeRef::Lazy(hash) => source.get(hash),
            };
            walk(&next, source, add(depth, 1), totals);
        }
    }

    fn collect_reachable(node: &Node, source: &mut Source<'_>, reachable: &mut HashSet<[u8; 32]>) {
        if !reachable.insert(node.structural_hash) {
            return;
        }
        for child in &node.children {
            let next = match child {
                NodeRef::Resolved(resolved) => Arc::clone(resolved),
                NodeRef::Lazy(hash) => source.get(hash),
            };
            collect_reachable(&next, source, reachable);
        }
    }

    struct Fixture {
        baseline: Vec<(String, String, String)>,
        base_root: Node,
        base_nodes: Store,
        root_record_len: usize,
    }

    fn fixture() -> Fixture {
        let baseline: Vec<(String, String, String)> = [
            ("m.room.create", ""),
            ("m.room.power_levels", ""),
            ("m.room.join_rules", ""),
            ("m.room.member", "@alice:example.org"),
            ("m.room.member", "@bob:example.org"),
        ]
        .iter()
        .enumerate()
        .map(|(i, (kind, key))| ((*kind).to_owned(), (*key).to_owned(), event_id(i)))
        .collect();
        let base_root = rezzy::hamt::build_hamt(
            ROOM.as_bytes(),
            baseline
                .iter()
                .map(|(kind, key, event)| (state_hamt_leaf_key(kind, key), event.clone())),
        )
        .unwrap();
        let built = build_state_hamt(ROOM, &room_hamt_prefix(ROOM), &baseline).unwrap();
        assert_eq!(base_root.structural_hash, built.root_hash);
        Fixture {
            baseline,
            base_root,
            base_nodes: built.nodes.iter().cloned().collect(),
            root_record_len: built.root_record.len(),
        }
    }

    fn tuples_for(mode: &str, mix: usize, events: usize) -> Vec<Tuple> {
        let mut tuples = Vec::new();
        for event in 0..events {
            let flagged = mix > 0 && event % 100 < mix;
            let disposition = match (flagged, mode) {
                (true, "reject") => 2,
                (true, "softfail") => 1,
                _ => 0,
            };
            tuples.push((
                "m.space.child".to_owned(),
                child_state_key(event),
                event_id(add(event, 10)),
                disposition,
            ));
        }
        tuples
    }

    /// Apply the chain incrementally, check it against full builds, and print
    /// the per-event node and delta bytes. Returns the steps and the store.
    fn chain_row(
        fx: &Fixture,
        tuples: &[Tuple],
        events: usize,
        label: &str,
    ) -> (Vec<rezzy::hamt::ChainStep<StateHamtKey, String>>, Store) {
        let applied = tuples.iter().filter(|tuple| tuple.3 != 2).count();
        let mut unresolved = |_: &[u8; 32]| -> Result<Node, ()> { Err(()) };
        let steps = rezzy::hamt::persist_chain(
            &fx.base_root,
            ROOM.as_bytes(),
            mutations(&decode_persisted(tuples)),
            &mut unresolved,
        )
        .unwrap();

        let mut store = fx.base_nodes.clone();
        let base_bytes: usize = store.values().map(Vec::len).sum();
        let (mut logical, mut created_nodes) = (0usize, 0usize);
        for step in &steps {
            for (hash, bytes) in &step.created {
                logical = add(logical, bytes.len());
                created_nodes = add(created_nodes, 1);
                store.entry(*hash).or_insert_with(|| bytes.clone());
            }
        }
        let unique = sub(store.values().map(Vec::len).sum::<usize>(), base_bytes);
        let delta_bytes: usize = tuples
            .iter()
            .map(|tuple| {
                let encoded = encode_delta(tuple);
                let mut decoded = decode_delta(&encoded);
                decoded.3 = tuple.3;
                assert_eq!(&decoded, tuple);
                encoded.len()
            })
            .sum();
        let verdict_bytes: usize = tuples
            .iter()
            .map(|tuple| persisted_verdict(tuple).len())
            .sum();

        let applied_tuples: Vec<_> = tuples.iter().filter(|tuple| tuple.3 != 2).collect();
        for sample in 1..=8usize {
            let count = div(mul(applied, sample), 8).max(1);
            let mut entries = fx.baseline.clone();
            entries.extend(
                applied_tuples[..count]
                    .iter()
                    .map(|(k, s, e, _)| ((*k).clone(), (*s).clone(), (*e).clone())),
            );
            let full = build_state_hamt(ROOM, &room_hamt_prefix(ROOM), &entries).unwrap();
            assert_eq!(
                full.root_hash,
                step_at(&steps, count).root_hash,
                "root at {count}"
            );
            for (hash, bytes) in &full.nodes {
                assert_eq!(store.get(hash), Some(bytes), "node bytes at {count}");
            }
        }

        let mut source = Source::new(&store);
        let mut totals = (0usize, 0usize);
        walk(&step_at(&steps, applied).root, &mut source, 1, &mut totals);
        assert_eq!(totals.0, add(fx.baseline.len(), applied));
        let mut snapshots = Vec::with_capacity(events);
        for event_count in 0..events {
            let applied_at_event = applied_through(tuples, add(event_count, 1));
            snapshots.push(if applied_at_event == 0 {
                Arc::clone(&fx.base_root)
            } else {
                Arc::clone(&step_at(&steps, applied_at_event).root)
            });
        }
        let mut snapshot_logical_bytes = 0usize;
        let mut snapshot_unique_nodes = HashSet::new();
        for root in snapshots {
            let mut reachable = HashSet::new();
            collect_reachable(&root, &mut source, &mut reachable);
            for hash in reachable {
                snapshot_logical_bytes = add(
                    snapshot_logical_bytes,
                    store.get(&hash).expect("snapshot node stored").len(),
                );
                snapshot_unique_nodes.insert(hash);
            }
        }
        let snapshot_unique_bytes: usize = snapshot_unique_nodes
            .iter()
            .map(|hash| store.get(hash).expect("unique node stored").len())
            .sum();
        let snapshot_shared_pct = tenths(
            mul(
                snapshot_logical_bytes.saturating_sub(snapshot_unique_bytes),
                100,
            ),
            snapshot_logical_bytes,
        );
        println!(
            "{label}\t{applied}\t{}\t{}\t{}\t{snapshot_shared_pct}\t{}\t{}\t{}",
            tenths(logical, events),
            tenths(unique, events),
            tenths(created_nodes, events),
            totals.1,
            tenths(delta_bytes, events),
            tenths(verdict_bytes, events),
        );
        (steps, store)
    }

    /// Storage when only every K-th root is materialized.
    fn keyframe_rows(
        fx: &Fixture,
        tuples: &[Tuple],
        steps: &[rezzy::hamt::ChainStep<StateHamtKey, String>],
        store: &Store,
        events: usize,
    ) {
        println!("K\tcheckpoints\tstgp_B_ev\troot_rec_B_ev\tkey_stat_B_ev\tdir_total_B\tdir_B_ev");
        for k in KEYFRAME_SWEEP {
            let mut seen: HashSet<[u8; 32]> = fx.base_nodes.keys().copied().collect();
            let (mut stat_bytes, mut checkpoints) = (0usize, 0usize);
            let mut checkpoint_groups = HashSet::new();
            for event_count in (k..=events).step_by(k) {
                checkpoints = add(checkpoints, 1);
                let applied = applied_through(tuples, event_count);
                if applied == 0 {
                    continue;
                }
                checkpoint_groups.insert(applied);
                let mut stack = vec![step_at(steps, applied).root_hash];
                while let Some(hash) = stack.pop() {
                    if !seen.insert(hash) {
                        continue;
                    }
                    let bytes = store.get(&hash).expect("reachable node stored");
                    stat_bytes = add(stat_bytes, bytes.len());
                    let node = PersistedInternalNode::<StateHamtKey, String>::decode_v1_verified(
                        bytes,
                        ROOM.as_bytes(),
                        hash,
                    )
                    .unwrap();
                    stack.extend(node.children.iter().map(NodeRef::structural_hash));
                }
            }
            println!(
                "{k}\t{checkpoints}\t{}\t{}\t{}\t{}\t{}",
                tenths(mul(checkpoint_groups.len(), STGP_PAYLOAD), events),
                tenths(mul(checkpoint_groups.len(), fx.root_record_len), events),
                tenths(stat_bytes, events),
                mul(checkpoints, DIRECTORY_ENTRY),
                hundredths(mul(checkpoints, DIRECTORY_ENTRY), events),
            );
        }
    }

    /// Write the same encoded STAT/STGP payload shapes through `PackfileStorage`
    /// and report actual collection-index and physical-frame accounting.
    #[allow(
        clippy::too_many_lines,
        reason = "keeps the measured put_many layout explicit"
    )]
    fn put_many_rows(
        fx: &Fixture,
        tuples: &[Tuple],
        steps: &[rezzy::hamt::ChainStep<StateHamtKey, String>],
        stored_nodes: &Store,
        k: usize,
        label: &str,
    ) {
        use mtxdb::{
            CollectionMetadata, DigestAlgorithm, FrameIdPolicy, PackfileStorage, PayloadPolicy,
            RecordIdentityRule, StateGroupDerivation, StateGroupInstance, StateGroupResolver,
            StateGroupTrust, StorageEngine,
        };

        crate::sweep_stale_test_dirs();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "mtxdb-state-keyframes-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let stat = state_hamt_collection_id(ROOM).unwrap();
        let stgp = mtxdb::state_group_collection_id(ROOM).unwrap();
        let keyf_canonical = format!("{ROOM}/keyframe-directory");
        let keyf =
            mtxdb::derive_group_member_collection_id(*b"STAT", keyf_canonical.as_bytes()).unwrap();
        let metadata = |namespace, canonical_id: &[u8]| CollectionMetadata {
            member_namespace: namespace,
            collection_canonical_id: canonical_id.to_vec(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::ExternalCanonicalIdToCrosscheck,
                digest_algorithm: DigestAlgorithm::Sha256,
            },
            payload: PayloadPolicy::Source,
            extension: None,
            role: Some("state-keyframe-measure".to_owned()),
            schema: Some("state-keyframe-measure.v1".to_owned()),
        };
        store
            .ensure_collection_metadata(&stat, &metadata(Some(*b"STAT"), ROOM.as_bytes()))
            .unwrap();
        store
            .ensure_collection_metadata(&stgp, &metadata(Some(*b"STGP"), ROOM.as_bytes()))
            .unwrap();
        if !tuples.is_empty() {
            store
                .ensure_collection_metadata(
                    &keyf,
                    &metadata(Some(*b"STAT"), keyf_canonical.as_bytes()),
                )
                .unwrap();
        }

        let mut stat_records = Vec::new();
        let mut root_hashes = vec![fx.base_root.structural_hash];
        let mut lattice = LtHash::default();
        for (kind, key, event) in &fx.baseline {
            lattice.insert(kind, key, event);
        }
        let base_id = rezzy::hamt::state_group_id_from_lthash(&lattice);
        let base_instance = mtxdb::state_group_instance_id(
            ROOM,
            &[],
            &base_id,
            StateGroupDerivation::new(StateGroupResolver::V2, StateGroupTrust::TrustedImport),
        );
        let base_root_id = state_hamt_root_node_id(ROOM, &base_id);
        stat_records.push((
            base_root_id,
            mtxdb::NodeData::from_slice(&encode_state_hamt_root(
                &room_hamt_prefix(ROOM),
                ROOM,
                &fx.base_root.structural_hash,
                &lattice,
            )),
        ));

        let mut stgp_records = Vec::new();
        let mut keyf_records = Vec::new();
        let base_record = StateGroupInstance {
            parents: Vec::new(),
            lthash: base_id,
            root_id: base_root_id,
            derivation: StateGroupDerivation::new(
                StateGroupResolver::V2,
                StateGroupTrust::TrustedImport,
            ),
        };
        stgp_records.push((
            base_instance,
            mtxdb::NodeData::from_slice(&mtxdb::encode_state_group_record(&base_record)),
        ));

        let mut parent = base_instance;
        let mut applied = 0usize;
        for (event_index, tuple) in tuples.iter().enumerate() {
            if tuple.3 == 2 {
                continue;
            }
            applied = add(applied, 1);
            let event_count = add(event_index, 1);
            lattice.insert(&tuple.0, &tuple.1, &tuple.2);
            if event_count.checked_rem(k).unwrap() != 0 {
                continue;
            }
            let lthash = rezzy::hamt::state_group_id_from_lthash(&lattice);
            let root_hash = step_at(steps, applied).root_hash;
            root_hashes.push(root_hash);
            let root_id = state_hamt_root_node_id(ROOM, &lthash);
            stat_records.push((
                root_id,
                mtxdb::NodeData::from_slice(&encode_state_hamt_root(
                    &room_hamt_prefix(ROOM),
                    ROOM,
                    &root_hash,
                    &lattice,
                )),
            ));
            let derivation =
                StateGroupDerivation::new(StateGroupResolver::V2, StateGroupTrust::TrustedImport);
            let instance_id = mtxdb::state_group_instance_id(ROOM, &[parent], &lthash, derivation);
            let record = StateGroupInstance {
                parents: vec![parent],
                lthash,
                root_id,
                derivation,
            };
            stgp_records.push((
                instance_id,
                mtxdb::NodeData::from_slice(&mtxdb::encode_state_group_record(&record)),
            ));
            parent = instance_id;
            let checkpoint_id = mtxdb::event_record_id(tuple.2.as_bytes());
            let mut directory_value = Vec::with_capacity(DIRECTORY_ENTRY);
            directory_value.extend_from_slice(&root_id);
            directory_value.extend_from_slice(&u32::try_from(event_count).unwrap().to_be_bytes());
            keyf_records.push((checkpoint_id, mtxdb::NodeData::from_slice(&directory_value)));
        }
        // Persist only the nodes reachable from keyframe roots. This is the
        // actual content-addressed STAT union for this selected layout.
        let mut node_ids = HashSet::new();
        for root_hash in root_hashes {
            let mut stack = vec![root_hash];
            while let Some(hash) = stack.pop() {
                if !node_ids.insert(hash) {
                    continue;
                }
                let bytes = stored_nodes.get(&hash).expect("keyframe node exists");
                let mut id = [0; 16];
                id.copy_from_slice(&hash[..16]);
                stat_records.push((id, mtxdb::NodeData::from_slice(bytes)));
                let node = PersistedInternalNode::<StateHamtKey, String>::decode_v1_verified(
                    bytes,
                    ROOM.as_bytes(),
                    hash,
                )
                .unwrap();
                stack.extend(node.children.iter().map(NodeRef::structural_hash));
            }
        }
        store.put_many(&stat, &stat_records).unwrap();
        store.put_many(&stgp, &stgp_records).unwrap();
        if !keyf_records.is_empty() {
            store.put_many(&keyf, &keyf_records).unwrap();
        }
        store.sync_all().unwrap();
        if k == 1 && !tuples.is_empty() {
            direct_packfile_reads(&store, &stat, &keyf, tuples);
        }
        let (stat_n, stat_index, _) = store.collection_index_info(&stat).unwrap();
        let (stgp_n, stgp_index, _) = store.collection_index_info(&stgp).unwrap();
        let (keyf_n, keyf_index) = store
            .collection_index_info(&keyf)
            .map_or((0, 0), |(count, index, _)| (count, index));
        let disk = PackfileStorage::collection_disk_bytes_from_disk(&dir).unwrap();
        println!(
            "# put_many layout={label} STAT records={stat_n} index_B={stat_index} frame_B={} frame_B_ev={} index_B_ev={} STGP records={stgp_n} index_B={stgp_index} frame_B={} frame_B_ev={} index_B_ev={} KEYF records={keyf_n} index_B={keyf_index} frame_B={} frame_B_ev={} index_B_ev={}",
            disk.get(&stat).copied().unwrap_or_default(),
            if tuples.is_empty() { "-".to_owned() } else { tenths(usize::try_from(disk.get(&stat).copied().unwrap_or_default()).unwrap(), tuples.len()) },
            if tuples.is_empty() { "-".to_owned() } else { tenths(stat_index, tuples.len()) },
            disk.get(&stgp).copied().unwrap_or_default(),
            if tuples.is_empty() { "-".to_owned() } else { tenths(usize::try_from(disk.get(&stgp).copied().unwrap_or_default()).unwrap(), tuples.len()) },
            if tuples.is_empty() { "-".to_owned() } else { tenths(stgp_index, tuples.len()) },
            disk.get(&keyf).copied().unwrap_or_default(),
            if tuples.is_empty() { "-".to_owned() } else { tenths(usize::try_from(disk.get(&keyf).copied().unwrap_or_default()).unwrap(), tuples.len()) },
            if tuples.is_empty() { "-".to_owned() } else { tenths(keyf_index, tuples.len()) },
        );
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn direct_packfile_reads(
        store: &mtxdb::PackfileStorage,
        stat: &[u8; 16],
        keyf: &[u8; 16],
        tuples: &[Tuple],
    ) {
        use mtxdb::StorageEngine;

        let targets = sample_targets(tuples.len());
        for shape in ["root", "entries"] {
            for cache in ["cold", "hot"] {
                let mut hot_decoded = HashMap::<[u8; 16], Node>::new();
                let mut timings = Vec::new();
                let passes = if cache == "cold" { 1 } else { 2 };
                for pass in 0..passes {
                    for &target in &targets {
                        let event_node =
                            mtxdb::event_record_id(tuples[sub(target, 1)].2.as_bytes());
                        let started = Instant::now();
                        let directory = store
                            .get_many_bytes(keyf, &[event_node])
                            .unwrap()
                            .pop()
                            .flatten()
                            .expect("direct event directory record");
                        let root_id: [u8; 16] = directory[..16].try_into().unwrap();
                        let root_record = store
                            .get_many_bytes(stat, &[root_id])
                            .unwrap()
                            .pop()
                            .flatten()
                            .expect("direct state root record");
                        let root_hash = decode_root_hash(&root_record);
                        if shape == "entries" {
                            let mut cold_decoded = HashMap::new();
                            let decoded = if cache == "hot" {
                                &mut hot_decoded
                            } else {
                                &mut cold_decoded
                            };
                            let root = packfile_node(store, stat, &root_hash, decoded);
                            let mut seen = HashSet::new();
                            let count =
                                packfile_entry_count(store, stat, &root, decoded, &mut seen);
                            assert_eq!(count, add(5, target));
                        }
                        if cache == "cold" || pass == 1 {
                            timings.push(started.elapsed().as_nanos());
                        }
                    }
                }
                timings.sort_unstable();
                let pick = |quantile: usize| {
                    timings[div(mul(timings.len().saturating_sub(1), quantile), 100)]
                };
                println!(
                    "# direct_packfile_read shape={shape} cache={cache} p50_ns={} p99_ns={} samples={}",
                    pick(50),
                    pick(99),
                    timings.len(),
                );
            }
        }
    }

    fn decode_root_hash(record: &[u8]) -> [u8; 32] {
        assert!(record.starts_with(b"MTHR\x01"));
        let prefix_len = usize::from(u16::from_be_bytes([record[5], record[6]]));
        let room_len_at = add(7, prefix_len);
        let room_len = usize::from(u16::from_be_bytes([
            record[room_len_at],
            record[add(room_len_at, 1)],
        ]));
        let hash_at = add(add(room_len_at, 2), room_len);
        record[hash_at..add(hash_at, 32)].try_into().unwrap()
    }

    fn packfile_node(
        store: &mtxdb::PackfileStorage,
        stat: &[u8; 16],
        hash: &[u8; 32],
        cache: &mut HashMap<[u8; 16], Node>,
    ) -> Node {
        use mtxdb::StorageEngine;

        let mut id = [0; 16];
        id.copy_from_slice(&hash[..16]);
        if let Some(node) = cache.get(&id) {
            return Arc::clone(node);
        }
        let bytes = store
            .get_many_bytes(stat, &[id])
            .unwrap()
            .pop()
            .flatten()
            .expect("direct STAT node record");
        let node = Arc::new(
            PersistedInternalNode::<StateHamtKey, String>::decode_v1_verified(
                &bytes,
                ROOM.as_bytes(),
                *hash,
            )
            .unwrap(),
        );
        cache.insert(id, Arc::clone(&node));
        node
    }

    fn packfile_entry_count(
        store: &mtxdb::PackfileStorage,
        stat: &[u8; 16],
        node: &Node,
        cache: &mut HashMap<[u8; 16], Node>,
        seen: &mut HashSet<[u8; 32]>,
    ) -> usize {
        if !seen.insert(node.structural_hash) {
            return 0;
        }
        let mut count = node.leaves.len();
        for child in &node.children {
            let hash = child.structural_hash();
            let next = packfile_node(store, stat, &hash, cache);
            count = add(count, packfile_entry_count(store, stat, &next, cache, seen));
        }
        count
    }

    /// Replay from the nearest keyframe to sampled historical events.
    fn replay_rows(
        fx: &Fixture,
        tuples: &[Tuple],
        steps: &[rezzy::hamt::ChainStep<StateHamtKey, String>],
        store: &Store,
    ) {
        println!("K\tshape\tcache\tp50_us\tp99_us\tinserts_avg\tnode_reads_avg");
        let context = ReplayContext::new(fx, tuples, steps);
        let targets = sample_targets(tuples.len());
        // Baseline without keyframes or the mutator: rebuild from all entries.
        let mut rebuild = Vec::new();
        for &target in &targets {
            let applied = applied_through(tuples, target);
            let mut entries = fx.baseline.clone();
            entries.extend(
                tuples[..target]
                    .iter()
                    .filter(|tuple| tuple.3 != 2)
                    .map(|(kind, key, event, _)| (kind.clone(), key.clone(), event.clone())),
            );
            let started = Instant::now();
            let built = build_state_hamt(ROOM, &room_hamt_prefix(ROOM), &entries).unwrap();
            rebuild.push(started.elapsed().as_micros());
            assert_eq!(built.root_hash, root_hash_at(fx, steps, applied));
        }
        rebuild.sort_unstable();
        println!(
            "no_stored_state_full_rebuild\troot\tfull\t{}\t{}\t{}\t-",
            rebuild[div(mul(rebuild.len().saturating_sub(1), 50), 100)],
            rebuild[div(mul(rebuild.len().saturating_sub(1), 99), 100)],
            applied_through(tuples, tuples.len()),
        );
        for k in std::iter::once(0usize).chain(KEYFRAME_SWEEP) {
            for shape in ["root", "entries"] {
                for cache in ["cold", "hot"] {
                    let mut hot = Source::new(store);
                    if cache == "hot" {
                        for &target in &targets {
                            context.once(&mut hot, k, target, shape);
                        }
                    }
                    let (mut micros, mut inserts, mut reads) = (Vec::new(), 0usize, 0usize);
                    for &target in &targets {
                        let mut cold = Source::new(store);
                        let source = if cache == "hot" { &mut hot } else { &mut cold };
                        let before = source.reads;
                        let started = Instant::now();
                        context.once(source, k, target, shape);
                        micros.push(started.elapsed().as_micros());
                        let frame = if k == 0 {
                            target
                        } else {
                            mul(div(target, k), k)
                        };
                        inserts = add(
                            inserts,
                            sub(
                                context.applied_by_event[target],
                                context.applied_by_event[frame],
                            ),
                        );
                        reads = add(reads, sub(source.reads, before));
                    }
                    micros.sort_unstable();
                    let pick = |quantile: usize| {
                        micros[div(mul(micros.len().saturating_sub(1), quantile), 100)]
                    };
                    println!(
                        "{}\t{shape}\t{cache}\t{}\t{}\t{}\t{}",
                        if k == 0 {
                            "direct".to_owned()
                        } else {
                            k.to_string()
                        },
                        pick(50),
                        pick(99),
                        tenths(inserts, targets.len()),
                        tenths(reads, targets.len()),
                    );
                }
            }
        }
        entry_walk_rows(&context, store, &targets);
    }

    fn sample_targets(event_count: usize) -> Vec<usize> {
        (1..=48usize)
            .map(|i| div(mul(event_count, i), 48).max(1))
            .collect()
    }

    struct ReplayContext<'a> {
        fx: &'a Fixture,
        tuples: &'a [Tuple],
        steps: &'a [rezzy::hamt::ChainStep<StateHamtKey, String>],
        applied_by_event: Vec<usize>,
    }

    impl<'a> ReplayContext<'a> {
        fn new(
            fx: &'a Fixture,
            tuples: &'a [Tuple],
            steps: &'a [rezzy::hamt::ChainStep<StateHamtKey, String>],
        ) -> Self {
            let mut applied_by_event = Vec::with_capacity(add(tuples.len(), 1));
            applied_by_event.push(0);
            for tuple in tuples {
                let next = add(
                    *applied_by_event.last().expect("initial count exists"),
                    usize::from(tuple.3 != 2),
                );
                applied_by_event.push(next);
            }
            Self {
                fx,
                tuples,
                steps,
                applied_by_event,
            }
        }

        fn once(&self, source: &mut Source<'_>, k: usize, target: usize, shape: &str) {
            let target_applied = self.applied_by_event[target];
            if k == 0 {
                let direct_hash = root_hash_at(self.fx, self.steps, target_applied);
                let root = if target_applied == 0 {
                    Arc::clone(&self.fx.base_root)
                } else {
                    source.get(&direct_hash)
                };
                if shape == "entries" {
                    let mut totals = (0usize, 0usize);
                    walk(&root, source, 1, &mut totals);
                    assert_eq!(totals.0, add(self.fx.baseline.len(), target_applied));
                }
                return;
            }
            self.replay_from_keyframe(source, k, target, target_applied, shape);
        }

        fn replay_from_keyframe(
            &self,
            source: &mut Source<'_>,
            k: usize,
            target: usize,
            target_applied: usize,
            shape: &str,
        ) {
            let frame_event = mul(div(target, k), k);
            let frame_applied = self.applied_by_event[frame_event];
            let root = if frame_applied == 0 {
                Arc::clone(&self.fx.base_root)
            } else {
                source.get(&step_at(self.steps, frame_applied).root_hash)
            };
            let persisted_deltas = decode_persisted(&self.tuples[frame_event..target]);
            let deltas = mutations(&persisted_deltas);
            let mut resolve = |hash: &[u8; 32]| -> Result<Node, ()> { Ok(source.get(hash)) };
            let chain =
                rezzy::hamt::persist_chain(&root, ROOM.as_bytes(), deltas, &mut resolve).unwrap();
            let tip = chain.last().map_or(root, |step| Arc::clone(&step.root));
            if shape == "entries" {
                let mut totals = (0usize, 0usize);
                walk(&tip, source, 1, &mut totals);
                assert_eq!(totals.0, add(self.fx.baseline.len(), target_applied));
            }
            assert_eq!(
                tip.structural_hash,
                root_hash_at(self.fx, self.steps, target_applied),
                "replay root at {target}"
            );
        }
    }

    fn entry_walk_rows(context: &ReplayContext<'_>, store: &Store, targets: &[usize]) {
        for cache in ["cold", "hot"] {
            let mut hot = Source::new(store);
            if cache == "hot" {
                for &target in targets {
                    let applied = context.applied_by_event[target];
                    let root_hash = root_hash_at(context.fx, context.steps, applied);
                    let root = hot.get(&root_hash);
                    let mut totals = (0, 0);
                    walk(&root, &mut hot, 1, &mut totals);
                }
            }
            let (mut micros, mut reads) = (Vec::new(), 0usize);
            for &target in targets {
                let mut cold = Source::new(store);
                let source = if cache == "hot" { &mut hot } else { &mut cold };
                let root_hash =
                    root_hash_at(context.fx, context.steps, context.applied_by_event[target]);
                let root = source.get(&root_hash);
                let before = source.reads;
                let started = Instant::now();
                let mut totals = (0, 0);
                walk(&root, source, 1, &mut totals);
                assert_eq!(
                    totals.0,
                    add(context.fx.baseline.len(), context.applied_by_event[target])
                );
                micros.push(started.elapsed().as_micros());
                reads = add(reads, sub(source.reads, before));
            }
            micros.sort_unstable();
            let pick =
                |quantile: usize| micros[div(mul(micros.len().saturating_sub(1), quantile), 100)];
            println!(
                "walk_only_{cache}\tentries\t{cache}\t{}\t{}\t0\t{}",
                pick(50),
                pick(99),
                tenths(reads, targets.len()),
            );
        }
    }

    fn applied_through(tuples: &[Tuple], event_count: usize) -> usize {
        tuples[..event_count]
            .iter()
            .filter(|tuple| tuple.3 != 2)
            .count()
    }

    fn root_hash_at(
        fx: &Fixture,
        steps: &[rezzy::hamt::ChainStep<StateHamtKey, String>],
        applied: usize,
    ) -> [u8; 32] {
        if applied == 0 {
            fx.base_root.structural_hash
        } else {
            step_at(steps, applied).root_hash
        }
    }

    /// Linear-space measurement for keyframe + delta state storage.
    ///
    /// Not a correctness test of the CLI: run with
    /// `cargo test --release -- --ignored --nocapture state_keyframe_measure`.
    /// `MEASURE_EVENTS` overrides the event count (default 4096).
    ///
    /// The state fixture is a baseline (create, power levels, join rules, two
    /// members) followed by one unique `m.space.child` tuple per event, using
    /// canonical-shaped event IDs. Each applicable event is applied with
    /// `persist_chain` (one tuple on the previous root); emitted nodes are the
    /// logical path copies. At sampled
    /// counts the incremental root and every node of a full
    /// `build_state_hamt` must match byte for byte, and every replayed root
    /// must equal the recorded one. Each delta stores type, state key and
    /// event ID as length-prefixed UTF-8; the separately encoded persisted
    /// `Verdict` gates application. Rejected applies nothing; soft-failed
    /// applies in the federated view. STAT/STGP payloads are written through
    /// `PackfileStorage::put_many`; the fixture does not run a full PDU import.
    /// Replay checks every disposition mix.
    #[test]
    #[ignore = "measurement, not a correctness test"]
    fn state_keyframe_measure() {
        let events: usize = std::env::var("MEASURE_EVENTS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(4096);
        let fx = fixture();
        println!(
            "# baseline: create=1 PL=1 join_rules=1 members=2; root_record {} B, unique STAT {} B ({} nodes), S0 = {}",
            fx.root_record_len,
            fx.base_nodes.values().map(Vec::len).sum::<usize>(),
            fx.base_nodes.len(),
            fx.baseline.len(),
        );
        for (label, kind) in [
            ("create", "m.room.create"),
            ("power_levels", "m.room.power_levels"),
            ("join_rules", "m.room.join_rules"),
            ("members", "m.room.member"),
        ] {
            let bytes = fx
                .baseline
                .iter()
                .filter(|(event_type, _, _)| event_type == kind)
                .map(|(event_type, state_key, event_id)| {
                    encode_delta(&(event_type.clone(), state_key.clone(), event_id.clone(), 0))
                        .len()
                })
                .sum::<usize>();
            println!("# baseline {label} replay-delta-bytes={bytes}");
        }
        println!(
            "row\tapplied\tlogical_B_ev\tunique_B_ev\tnodes_ev\tshared_pct\tdepth\tdelta_B_ev\tverdict_B_ev"
        );
        for (mode, mix) in [
            ("none", 0usize),
            ("reject", 1),
            ("reject", 10),
            ("softfail", 1),
            ("softfail", 10),
        ] {
            let tuples = tuples_for(mode, mix, events);
            let (steps, store) = chain_row(&fx, &tuples, events, &format!("{mode}{mix}%"));
            if mix == 0 {
                put_many_rows(&fx, &[], &[], &fx.base_nodes, 1, "baseline-only");
                for k in std::iter::once(1usize).chain(KEYFRAME_SWEEP) {
                    let label = if k == 1 {
                        "direct".to_owned()
                    } else {
                        format!("K={k}")
                    };
                    put_many_rows(&fx, &tuples, &steps, &store, k, &label);
                }
            }
            keyframe_rows(&fx, &tuples, &steps, &store, events);
            replay_rows(&fx, &tuples, &steps, &store);
        }
    }
}
