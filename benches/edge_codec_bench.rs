//! Edge-encoding benchmark for a synthetic Matrix room DAG.
//!
//! Compares space and lookup cost of forward (prev/auth) and reverse
//! (children-of, relations) encodings over dense room-local u32 short ids,
//! plus head/sealed rewrite amplification under batched ingest.
//!
//! Usage: edgebench [events] [seed]

use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

// ---------------------------------------------------------------- rng
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

// ---------------------------------------------------------------- model
struct Room {
    prev: Vec<Vec<u32>>,         // forward, per child short
    auth: Vec<Vec<u32>>,         // forward, per child short
    rel: Vec<Option<(u32, u8)>>, // forward relation (parent, rel_type)
}

/// Matrix-shaped room: short 0 = create, 1 = join_rules, power levels
/// re-issued every ~50k events, one member event per user on first send,
/// skewed senders, prev to recent events (rare forks), ~15% relations
/// (reactions/edits/replies) skewed toward recent and a hot set.
fn gen_room(n: usize, seed: u64) -> Room {
    let mut r = Rng(seed | 1);
    let users = 2000u64;
    let mut prev = vec![Vec::new(); n];
    let mut auth = vec![Vec::new(); n];
    let mut rel = vec![None; n];
    let mut member: HashMap<u64, u32> = HashMap::new();
    let mut pl: u32 = 1;
    let mut hot: Vec<u32> = Vec::new();
    for s in 2..n as u32 {
        let i = s as usize;
        // prev: 1 (92%), 2 (7%), 3..5 (1%)
        let roll = r.f();
        let k = if roll < 0.92 { 1 } else if roll < 0.99 { 2 } else { 3 + r.below(3) as usize };
        let mut ps = Vec::with_capacity(k);
        for _ in 0..k {
            let back = 1 + r.below(8.min(s as u64 - 1)) as u32;
            let p = s - back;
            if !ps.contains(&p) {
                ps.push(p);
            }
        }
        ps.sort_unstable();
        prev[i] = ps;

        let u = ((r.f() * r.f()) * users as f64) as u64; // skewed sender
        let mut a = vec![0u32, pl];
        let is_member = !member.contains_key(&u);
        if is_member {
            a.push(1); // join rules
            member.insert(u, s);
        } else {
            a.push(member[&u]);
        }
        if s % 50_000 == 0 {
            pl = s; // this event is a new power-levels event
        }
        a.sort_unstable();
        a.dedup();
        auth[i] = a;

        if !is_member && r.f() < 0.15 {
            let parent = if r.f() < 0.7 || hot.is_empty() {
                s - 1 - r.below(50.min(s as u64 - 1)) as u32
            } else {
                hot[(r.f() * r.f() * hot.len() as f64) as usize]
            };
            let ty = (r.below(4)) as u8; // annotation/replace/thread/reference
            rel[i] = Some((parent, ty));
        }
        if r.f() < 0.001 {
            hot.push(s);
        }
    }
    Room { prev, auth, rel }
}

// ---------------------------------------------------------------- varint
fn varint_len(mut v: u32) -> usize {
    let mut n = 1;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}
fn put_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}
fn get_varint(b: &[u8], at: &mut usize) -> u32 {
    let mut v = 0u32;
    let mut shift = 0;
    loop {
        let x = b[*at];
        *at += 1;
        v |= ((x & 0x7f) as u32) << shift;
        if x < 0x80 {
            return v;
        }
        shift += 7;
    }
}

const CHUNK: usize = 1024;

// ---------------------------------------------------------------- forward encodings
/// F2: raw CSR, offsets u32[C+1] + targets u32.
struct CsrRaw {
    chunks: Vec<(Vec<u32>, Vec<u32>)>,
}
impl CsrRaw {
    fn build(lists: &[Vec<u32>]) -> Self {
        let chunks = lists
            .chunks(CHUNK)
            .map(|c| {
                let mut off = Vec::with_capacity(c.len() + 1);
                let mut tg = Vec::new();
                off.push(0);
                for l in c {
                    tg.extend_from_slice(l);
                    off.push(tg.len() as u32);
                }
                (off, tg)
            })
            .collect();
        Self { chunks }
    }
    fn bytes(&self) -> usize {
        self.chunks.iter().map(|(o, t)| 4 * (o.len() + t.len())).sum()
    }
    fn get(&self, s: u32) -> &[u32] {
        let (o, t) = &self.chunks[s as usize / CHUNK];
        let i = s as usize % CHUNK;
        &t[o[i] as usize..o[i + 1] as usize]
    }
}

/// F3: CSR with u16 byte offset per child, targets as varint(child - target)
/// for the first and varint(gap) after (targets sorted descending by delta).
struct CsrVar {
    chunks: Vec<(Vec<u16>, Vec<u8>)>,
}
impl CsrVar {
    fn build(lists: &[Vec<u32>]) -> Self {
        let mut chunks = Vec::new();
        for (ci, c) in lists.chunks(CHUNK).enumerate() {
            let mut off = Vec::with_capacity(c.len() + 1);
            let mut body = Vec::new();
            for (j, l) in c.iter().enumerate() {
                off.push(body.len() as u16);
                let child = (ci * CHUNK + j) as u32;
                // deltas ascending: nearest parent first
                let mut d: Vec<u32> = l.iter().map(|&p| child - p).collect();
                d.sort_unstable();
                let mut last = 0;
                for x in d {
                    put_varint(&mut body, x - last);
                    last = x;
                }
            }
            off.push(body.len() as u16);
            chunks.push((off, body));
        }
        Self { chunks }
    }
    fn bytes(&self) -> usize {
        // the trailing sentinel offset is the chunk length; count it
        self.chunks.iter().map(|(o, b)| 2 * o.len() + b.len()).sum()
    }
    fn get(&self, s: u32, out: &mut Vec<u32>) {
        out.clear();
        let (o, b) = &self.chunks[s as usize / CHUNK];
        let i = s as usize % CHUNK;
        let (mut at, end) = (o[i] as usize, o[i + 1] as usize);
        let mut acc = 0;
        while at < end {
            acc += get_varint(b, &mut at);
            out.push(s - acc);
        }
    }
}

/// F5: one roaring bitmap per chunk of children, key = (offset << 22) | delta.
/// Delta must fit 22 bits (rooms under ~4M events).
struct Roaring2d {
    chunks: Vec<RoaringBitmap>,
}
impl Roaring2d {
    fn build(lists: &[Vec<u32>]) -> Self {
        let chunks = lists
            .chunks(CHUNK)
            .enumerate()
            .map(|(ci, c)| {
                let mut b = RoaringBitmap::new();
                for (j, l) in c.iter().enumerate() {
                    let child = (ci * CHUNK + j) as u32;
                    for &p in l {
                        let d = child - p;
                        assert!(d < 1 << 22, "delta exceeds 22 bits");
                        b.insert(((j as u32) << 22) | d);
                    }
                }
                b
            })
            .collect();
        Self { chunks }
    }
    fn bytes(&self) -> usize {
        self.chunks.iter().map(RoaringBitmap::serialized_size).sum()
    }
    fn get(&self, s: u32, out: &mut Vec<u32>) {
        out.clear();
        let b = &self.chunks[s as usize / CHUNK];
        let j = (s as usize % CHUNK) as u32;
        let lo = j << 22;
        let start = if lo == 0 { 0 } else { b.rank(lo - 1) };
        let end = b.rank(lo | ((1 << 22) - 1));
        for k in start..end {
            let v = b.select(k as u32).unwrap();
            out.push(s - (v & ((1 << 22) - 1)));
        }
    }
}

/// F4: a roaring bitmap per child (as if each child's edge set were a bitmap).
fn roaring_per_list_bytes(lists: &[Vec<u32>]) -> usize {
    lists
        .iter()
        .filter(|l| !l.is_empty())
        .map(|l| l.iter().copied().collect::<RoaringBitmap>().serialized_size())
        .sum()
}

// ---------------------------------------------------------------- reverse
fn invert(lists: &[Vec<u32>], n: usize) -> Vec<Vec<u32>> {
    let mut rev = vec![Vec::new(); n];
    for (c, l) in lists.iter().enumerate() {
        for &p in l {
            rev[p as usize].push(c as u32);
        }
    }
    rev
}

/// R4/R5: reverse CSR keyed by parent chunk, children as varint gaps;
/// parents with in-degree above `hub` move to a roaring bitmap instead.
struct RevAdaptive {
    csr: CsrGaps,
    hubs: HashMap<u32, RoaringBitmap>,
}
struct CsrGaps {
    chunks: Vec<(Vec<u32>, Vec<u8>)>,
}
impl RevAdaptive {
    fn build(rev: &[Vec<u32>], hub: usize) -> Self {
        let mut hubs = HashMap::new();
        let mut chunks = Vec::new();
        for (ci, c) in rev.chunks(CHUNK).enumerate() {
            let mut off = Vec::with_capacity(c.len() + 1);
            let mut body = Vec::new();
            for (j, l) in c.iter().enumerate() {
                off.push(body.len() as u32);
                let p = (ci * CHUNK + j) as u32;
                if l.len() > hub {
                    hubs.insert(p, l.iter().copied().collect::<RoaringBitmap>());
                    continue;
                }
                let mut last = p;
                for &x in l {
                    put_varint(&mut body, x - last);
                    last = x;
                }
            }
            off.push(body.len() as u32);
            chunks.push((off, body));
        }
        Self { csr: CsrGaps { chunks }, hubs }
    }
    fn bytes(&self) -> (usize, usize) {
        // u32 offsets; a real layout could use u16 with a chunk base.
        let a = self.csr.chunks.iter().map(|(o, b)| 4 * o.len() + b.len()).sum();
        let h = self.hubs.values().map(RoaringBitmap::serialized_size).sum();
        (a, h)
    }
    fn count(&self, p: u32) -> usize {
        if let Some(b) = self.hubs.get(&p) {
            return b.len() as usize;
        }
        let (o, b) = &self.csr.chunks[p as usize / CHUNK];
        let i = p as usize % CHUNK;
        let (mut at, end) = (o[i] as usize, o[i + 1] as usize);
        let mut n = 0;
        while at < end {
            black_box(get_varint(b, &mut at));
            n += 1;
        }
        n
    }
}

// ---------------------------------------------------------------- head/sealed WA
/// Bytes written to persist a forward CSR-varint column when the head chunk
/// is rewritten once per ingest batch and sealed at `head` events.
fn forward_wa(lists: &[Vec<u32>], head: usize, batch: usize) -> (usize, usize) {
    let mut written = 0usize;
    let mut final_bytes = 0usize;
    let mut cur = 0usize;
    let mut in_head = 0usize;
    for (s, l) in lists.iter().enumerate() {
        let mut sz = 2; // u16 offset
        let mut d: Vec<u32> = l.iter().map(|&p| s as u32 - p).collect();
        d.sort_unstable();
        let mut last = 0;
        for x in d {
            sz += varint_len(x - last);
            last = x;
        }
        cur += sz;
        in_head += 1;
        let batch_end = (s + 1) % batch == 0;
        if batch_end || in_head == head {
            written += cur; // rewrite the whole head
        }
        if in_head == head {
            final_bytes += cur;
            cur = 0;
            in_head = 0;
        }
    }
    final_bytes += cur;
    written += cur;
    (written, final_bytes)
}

/// Bytes written to persist reverse edges when every touched record is
/// rewritten once per batch. Records: one CSR-gap record per 1024-parent
/// chunk; parents above `hub` children get bitmap records split by
/// child >> 16 (only the newest container moves).
fn reverse_wa(lists: &[Vec<u32>], n: usize, hub: usize, batch: usize) -> (usize, usize) {
    let mut deg = vec![0u32; n];
    let mut last_child = vec![0u32; n];
    let mut chunk_bytes = vec![0usize; n.div_ceil(CHUNK)];
    let mut hub_sets: HashMap<(u32, u32), RoaringBitmap> = HashMap::new();
    let mut touched_chunks: Vec<usize> = Vec::new();
    let mut touched_hubs: Vec<(u32, u32)> = Vec::new();
    let mut written = 0usize;
    for (c, l) in lists.iter().enumerate() {
        let c = c as u32;
        for &p in l {
            let pi = p as usize;
            deg[pi] += 1;
            if deg[pi] as usize > hub {
                let key = (p, c >> 16);
                hub_sets.entry(key).or_default().insert(c);
                if !touched_hubs.contains(&key) {
                    touched_hubs.push(key);
                }
            } else {
                let gap = c - if deg[pi] == 1 { p } else { last_child[pi] };
                chunk_bytes[pi / CHUNK] += varint_len(gap);
                last_child[pi] = c;
                if !touched_chunks.contains(&(pi / CHUNK)) {
                    touched_chunks.push(pi / CHUNK);
                }
            }
        }
        if (c as usize + 1) % batch == 0 {
            for &k in &touched_chunks {
                written += chunk_bytes[k] + 4 * (CHUNK + 1);
            }
            for k in &touched_hubs {
                written += hub_sets[k].serialized_size();
            }
            touched_chunks.clear();
            touched_hubs.clear();
        }
    }
    let final_bytes = chunk_bytes.iter().map(|b| b + 4 * (CHUNK + 1)).sum::<usize>()
        + hub_sets.values().map(RoaringBitmap::serialized_size).sum::<usize>();
    (written, final_bytes)
}

// ---------------------------------------------------------------- timing
fn ns_per<F: FnMut(u32)>(queries: &[u32], mut f: F) -> f64 {
    let t = Instant::now();
    for &q in queries {
        f(q);
    }
    t.elapsed().as_nanos() as f64 / queries.len() as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0x5EED);
    let room = gen_room(n, seed);
    let rel_fwd: Vec<Vec<u32>> = room
        .rel
        .iter()
        .map(|r| r.map(|(p, _)| vec![p]).unwrap_or_default())
        .collect();

    let mut q = Rng(seed ^ 0xABCD);
    let queries: Vec<u32> = (0..200_000).map(|_| 2 + q.below(n as u64 - 2) as u32).collect();

    println!("events\t{n}");
    // assumed point-tuple overhead: ~40 B frame (header + collection/logical id)
    // + 23 B sidecar/run entry. Verify against the packfile frame format.
    const TUPLE: usize = 63;

    println!("\n# forward\nfamily\tencoding\tedges\ttuples\tpayload_B\tB_per_edge\tB_per_edge_incl_tuples\tlookup_ns");
    for (name, lists) in [("prev", &room.prev), ("auth", &room.auth), ("rel", &rel_fwd)] {
        let edges: usize = lists.iter().map(Vec::len).sum();
        let nonempty = lists.iter().filter(|l| !l.is_empty()).count();
        let row = |enc: &str, tuples: usize, payload: usize, ns: f64| {
            println!(
                "{name}\t{enc}\t{edges}\t{tuples}\t{payload}\t{:.2}\t{:.2}\t{:.1}",
                payload as f64 / edges as f64,
                (payload + tuples * TUPLE) as f64 / edges as f64,
                ns
            );
        };
        // F1: per-event tuple holding u32 shorts (MatrixAdjacency shape)
        let f1 = edges * 4;
        let ns1 = ns_per(&queries, |s| {
            black_box(lists[s as usize].as_slice());
        });
        row("tuple_per_event_u32", nonempty, f1, ns1);

        let raw = CsrRaw::build(lists);
        let nsr = ns_per(&queries, |s| {
            black_box(raw.get(s));
        });
        row("csr_raw_u32_c1024", raw.chunks.len(), raw.bytes(), nsr);

        let var = CsrVar::build(lists);
        let mut buf = Vec::new();
        let nsv = ns_per(&queries, |s| {
            var.get(s, &mut buf);
            black_box(&buf);
        });
        row("csr_varint_delta_c1024", var.chunks.len(), var.bytes(), nsv);

        let r2 = Roaring2d::build(lists);
        let ns2 = ns_per(&queries, |s| {
            r2.get(s, &mut buf);
            black_box(&buf);
        });
        row("roaring_2d_c1024", r2.chunks.len(), r2.bytes(), ns2);

        let rp = roaring_per_list_bytes(lists);
        row("roaring_per_event", nonempty, rp, f64::NAN);
    }

    println!("\n# reverse (children-of)\nfamily\tencoding\tedges\tparents\thubs\ttuples\tpayload_B\tB_per_edge\tB_per_edge_incl_tuples\tlookup_ns\thub_lookup_ns");
    for (name, lists) in [("prev", &room.prev), ("auth", &room.auth), ("rel", &rel_fwd)] {
        let rev = invert(lists, n);
        let edges: usize = rev.iter().map(Vec::len).sum();
        let parents = rev.iter().filter(|l| !l.is_empty()).count();
        let pq: Vec<u32> = queries.iter().copied().filter(|&p| !rev[p as usize].is_empty()).collect();

        // per-parent tuple (u32 child list), i.e. one record per parent
        let f1 = edges * 4;
        println!(
            "{name}\ttuple_per_parent_u32\t{edges}\t{parents}\t0\t{parents}\t{f1}\t{:.2}\t{:.2}\t-\t-",
            f1 as f64 / edges as f64,
            (f1 + parents * TUPLE) as f64 / edges as f64
        );
        // per-parent roaring bitmap record
        let rpb = roaring_per_list_bytes(&rev);
        println!(
            "{name}\troaring_per_parent\t{edges}\t{parents}\t0\t{parents}\t{rpb}\t{:.2}\t{:.2}\t-\t-",
            rpb as f64 / edges as f64,
            (rpb + parents * TUPLE) as f64 / edges as f64
        );
        for hub in [usize::MAX, 4096, 256] {
            let ra = RevAdaptive::build(&rev, hub);
            let (a, h) = ra.bytes();
            let tuples = ra.csr.chunks.len() + ra.hubs.len();
            let ns = ns_per(&pq, |p| {
                black_box(ra.count(p));
            });
            let hub_ns = if lists.len() > 0 && name == "auth" {
                let hq: Vec<u32> = vec![0; 200];
                ns_per(&hq, |p| {
                    black_box(ra.count(p));
                })
            } else {
                f64::NAN
            };
            let label = if hub == usize::MAX {
                "csr_gap_varint_c1024".to_string()
            } else {
                format!("adaptive_hub>{hub}")
            };
            println!(
                "{name}\t{label}\t{edges}\t{parents}\t{}\t{tuples}\t{}\t{:.2}\t{:.2}\t{ns:.1}\t{hub_ns:.0}",
                ra.hubs.len(),
                a + h,
                (a + h) as f64 / edges as f64,
                (a + h + tuples * TUPLE) as f64 / edges as f64
            );
        }
    }

    println!("\n# head/sealed write amplification (bytes written / final bytes)");
    println!("family\tside\thead_or_hub\tbatch\twritten_B\tfinal_B\tWA");
    for (name, lists) in [("prev", &room.prev), ("auth", &room.auth)] {
        for head in [64, 256, 1024] {
            for batch in [1, 16, 128] {
                let (w, f) = forward_wa(lists, head, batch);
                println!("{name}\tforward\thead={head}\t{batch}\t{w}\t{f}\t{:.1}", w as f64 / f as f64);
            }
        }
    }
    for (name, lists) in [("prev", &room.prev), ("auth", &room.auth), ("rel", &rel_fwd)] {
        for batch in [1, 16, 128] {
            let (w, f) = reverse_wa(lists, n, 4096, batch);
            println!("{name}\treverse\thub>4096\t{batch}\t{w}\t{f}\t{:.1}", w as f64 / f as f64);
        }
    }
}
