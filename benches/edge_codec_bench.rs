//! Edge-encoding benchmark for a synthetic Matrix room DAG.
//!
//! Compares space and lookup cost of forward (prev/auth) and reverse
//! (children-of, relations) encodings over dense room-local u32 short ids,
//! plus head/sealed rewrite amplification under batched ingest.
//!
//! Usage: `edge_codec_bench` [events] [seed]

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

use mtxdb::{BitmapSet, DomainTag};

#[path = "edge_sensitivity.rs"]
mod sensitivity;

const BENCH_DOMAIN: DomainTag = DomainTag::new([0u8; 16]);
const CHUNK_SIZE: usize = 1024;
const TUPLE_OVERHEAD_BYTES: usize = 63;
const TWO_POW_32: f64 = 4_294_967_296.0;

fn num(n: u64) -> f64 {
    let high = u32::try_from(n >> 32).unwrap_or(u32::MAX);
    let low = u32::try_from(n & 0xFFFF_FFFF).unwrap_or(u32::MAX);
    f64::from(high) * TWO_POW_32 + f64::from(low)
}

fn len_f64(n: usize) -> f64 {
    num(u64::try_from(n).unwrap_or(u64::MAX))
}

// ---------------------------------------------------------------- rng
struct PseudoRng(u64);

impl PseudoRng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn next_fraction(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        num(value) / 9_007_199_254_740_992.0 // 2^53
    }

    fn below(&mut self, upper_bound: u64) -> u64 {
        let divisor = upper_bound.max(1);
        self.next_u64()
            .checked_rem(divisor)
            .expect("non-zero divisor")
    }

    fn sample_quadratic(&mut self, bound: u64) -> u64 {
        let r1 = u128::from(u32::try_from(self.next_u64() >> 32).unwrap_or(0));
        let r2 = u128::from(u32::try_from(self.next_u64() >> 32).unwrap_or(0));
        let product = (r1.saturating_mul(r2)) >> 32;
        let scaled = (product.saturating_mul(u128::from(bound))) >> 32;
        u64::try_from(scaled).expect("scaled fits in u64")
    }
}

// ---------------------------------------------------------------- model
struct RoomGraph {
    prev: Vec<Vec<u32>>,
    auth: Vec<Vec<u32>>,
    relations: Vec<Option<(u32, u8)>>,
}

/// Matrix-shaped room: short 0 = create, 1 = `join_rules`, power levels
/// re-issued every ~50k events, one member event per user on first send,
/// skewed senders, prev to recent events (rare forks), ~15% relations
/// (reactions/edits/replies) skewed toward recent and a hot set.
fn generate_room(event_count: usize, seed: u64) -> RoomGraph {
    let mut rng = PseudoRng(seed | 1);
    let user_count = 2000u64;
    let mut prev = vec![Vec::new(); event_count];
    let mut auth = vec![Vec::new(); event_count];
    let mut relations = vec![None; event_count];
    let mut user_member_events: HashMap<u64, u32> = HashMap::new();
    let mut active_power_levels_event: u32 = 1;
    let mut hot_events: Vec<u32> = Vec::new();

    let max_short = u32::try_from(event_count).expect("event count fits in u32");
    for current_short in 2..max_short {
        let event_index = usize::try_from(current_short).expect("short fits in usize");
        let roll = rng.next_fraction();
        let prev_count = if roll < 0.92 {
            1
        } else if roll < 0.99 {
            2
        } else {
            let extra = usize::try_from(rng.below(3)).expect("extra fits in usize");
            3_usize.saturating_add(extra)
        };

        let mut parents = Vec::with_capacity(prev_count);
        for _ in 0..prev_count {
            let max_lookback = 8u64.min(u64::from(current_short).saturating_sub(1));
            let lookback = 1u32.saturating_add(
                u32::try_from(rng.below(max_lookback)).expect("lookback fits in u32"),
            );
            let parent_short = current_short.saturating_sub(lookback);
            if !parents.contains(&parent_short) {
                parents.push(parent_short);
            }
        }
        parents.sort_unstable();
        prev[event_index] = parents;

        let user_id = rng.sample_quadratic(user_count);

        let mut auth_list = vec![0u32, active_power_levels_event];
        let is_first_send = !user_member_events.contains_key(&user_id);
        if is_first_send {
            auth_list.push(1); // join rules
            user_member_events.insert(user_id, current_short);
        } else {
            auth_list.push(user_member_events[&user_id]);
        }

        if current_short.is_multiple_of(50_000) {
            active_power_levels_event = current_short;
        }
        auth_list.sort_unstable();
        auth_list.dedup();
        auth[event_index] = auth_list;

        if !is_first_send && rng.next_fraction() < 0.15 {
            let target_parent = if rng.next_fraction() < 0.7 || hot_events.is_empty() {
                let max_rel_lookback = 50u64.min(u64::from(current_short).saturating_sub(1));
                let rel_lookback =
                    u32::try_from(rng.below(max_rel_lookback)).expect("rel lookback fits in u32");
                current_short.saturating_sub(1).saturating_sub(rel_lookback)
            } else {
                let hot_count = u64::try_from(hot_events.len()).expect("fits in u64");
                let hot_sample = rng.sample_quadratic(hot_count);
                let hot_index = usize::try_from(hot_sample).expect("fits in usize");
                hot_events[hot_index]
            };
            let rel_type = u8::try_from(rng.below(4)).expect("rel type fits in u8");
            relations[event_index] = Some((target_parent, rel_type));
        }

        if rng.next_fraction() < 0.001 {
            hot_events.push(current_short);
        }
    }

    RoomGraph {
        prev,
        auth,
        relations,
    }
}

// ---------------------------------------------------------------- varint & zigzag
fn zigzag_encode(val: i64) -> u64 {
    let shifted = val.wrapping_shl(1);
    let mask = val.wrapping_shr(63);
    (shifted ^ mask).cast_unsigned()
}

fn zigzag_decode(val: u64) -> i64 {
    let unsigned_half = (val >> 1).cast_signed();
    if val & 1 == 1 {
        !unsigned_half
    } else {
        unsigned_half
    }
}

fn varint_length(mut value: u32) -> usize {
    let mut count = 1usize;
    while value >= 0x80 {
        value >>= 7;
        count = count.saturating_add(1);
    }
    count
}

fn varint_length_u64(mut value: u64) -> usize {
    let mut count = 1usize;
    while value >= 0x80 {
        value >>= 7;
        count = count.saturating_add(1);
    }
    count
}

fn append_varint(buffer: &mut Vec<u8>, mut value: u32) {
    while value >= 0x80 {
        let byte = u8::try_from(value & 0x7f).expect("fits in u8") | 0x80;
        buffer.push(byte);
        value >>= 7;
    }
    buffer.push(u8::try_from(value).expect("fits in u8"));
}

fn append_varint_u64(buffer: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        let byte = u8::try_from(value & 0x7f).expect("fits in u8") | 0x80;
        buffer.push(byte);
        value >>= 7;
    }
    buffer.push(u8::try_from(value).expect("fits in u8"));
}

fn read_varint(bytes: &[u8], cursor: &mut usize) -> u32 {
    let mut decoded = 0u32;
    let mut shift_bits = 0u32;
    loop {
        let raw_byte = bytes[*cursor];
        *cursor = cursor.saturating_add(1);
        decoded |= u32::from(raw_byte & 0x7f) << shift_bits;
        if raw_byte < 0x80 {
            return decoded;
        }
        shift_bits = shift_bits.saturating_add(7);
    }
}

fn read_varint_u64(bytes: &[u8], cursor: &mut usize) -> u64 {
    let mut decoded = 0u64;
    let mut shift_bits = 0u32;
    loop {
        let raw_byte = bytes[*cursor];
        *cursor = cursor.saturating_add(1);
        decoded |= u64::from(raw_byte & 0x7f) << shift_bits;
        if raw_byte < 0x80 {
            return decoded;
        }
        shift_bits = shift_bits.saturating_add(7);
    }
}

// ---------------------------------------------------------------- forward encodings
struct CsrRawForward {
    chunks: Vec<(Vec<u32>, Vec<u32>)>,
}

impl CsrRawForward {
    fn build(edge_lists: &[Vec<u32>]) -> Self {
        let chunks = edge_lists
            .chunks(CHUNK_SIZE)
            .map(|chunk_slice| {
                let mut offsets = Vec::with_capacity(chunk_slice.len().saturating_add(1));
                let mut targets = Vec::new();
                offsets.push(0);
                for target_list in chunk_slice {
                    targets.extend_from_slice(target_list);
                    let target_offset = u32::try_from(targets.len()).expect("offset fits in u32");
                    offsets.push(target_offset);
                }
                (offsets, targets)
            })
            .collect();
        Self { chunks }
    }

    fn total_bytes(&self) -> usize {
        self.chunks
            .iter()
            .map(|(offsets, targets)| {
                let word_count = offsets.len().saturating_add(targets.len());
                word_count.saturating_mul(4)
            })
            .sum()
    }

    fn lookup(&self, short_id: u32) -> &[u32] {
        let chunk_index = usize::try_from(short_id).expect("fits in usize") / CHUNK_SIZE;
        let (offsets, targets) = &self.chunks[chunk_index];
        let within_chunk = usize::try_from(short_id).expect("fits in usize") % CHUNK_SIZE;
        let start_pos = usize::try_from(offsets[within_chunk]).expect("fits in usize");
        let next_index = within_chunk.saturating_add(1);
        let end_pos = usize::try_from(offsets[next_index]).expect("fits in usize");
        &targets[start_pos..end_pos]
    }
}

struct CsrVarintForward {
    chunks: Vec<(Vec<u16>, Vec<u8>)>,
}

impl CsrVarintForward {
    fn build(edge_lists: &[Vec<u32>]) -> Self {
        let mut chunks = Vec::new();
        for (chunk_index, chunk_slice) in edge_lists.chunks(CHUNK_SIZE).enumerate() {
            let mut offsets = Vec::with_capacity(chunk_slice.len().saturating_add(1));
            let mut payload = Vec::new();
            for (offset_in_chunk, target_list) in chunk_slice.iter().enumerate() {
                offsets.push(u16::try_from(payload.len()).expect("offset fits in u16"));
                let base_short = u32::try_from(
                    chunk_index
                        .saturating_mul(CHUNK_SIZE)
                        .saturating_add(offset_in_chunk),
                )
                .expect("short fits in u32");

                if !target_list.is_empty() {
                    let mut sorted = target_list.clone();
                    sorted.sort_unstable_by(|a, b| b.cmp(a));
                    let diff_0 = i64::from(base_short)
                        .checked_sub(i64::from(sorted[0]))
                        .expect("valid diff");
                    append_varint_u64(&mut payload, zigzag_encode(diff_0));

                    for window in sorted.windows(2) {
                        let gap = window[0].checked_sub(window[1]).expect("sorted descending");
                        append_varint(&mut payload, gap);
                    }
                }
            }
            offsets.push(u16::try_from(payload.len()).expect("offset fits in u16"));
            chunks.push((offsets, payload));
        }
        Self { chunks }
    }

    fn total_bytes(&self) -> usize {
        self.chunks
            .iter()
            .map(|(offsets, payload)| {
                let offset_bytes = offsets.len().saturating_mul(2);
                offset_bytes.saturating_add(payload.len())
            })
            .sum()
    }

    fn lookup(&self, short_id: u32, output: &mut Vec<u32>) {
        output.clear();
        let short_usize = usize::try_from(short_id).expect("fits in usize");
        let (offsets, payload) = &self.chunks[short_usize / CHUNK_SIZE];
        let within_chunk = short_usize % CHUNK_SIZE;
        let mut cursor = usize::from(offsets[within_chunk]);
        let end_cursor = usize::from(offsets[within_chunk.saturating_add(1)]);

        if cursor < end_cursor {
            let initial_diff = zigzag_decode(read_varint_u64(payload, &mut cursor));
            let parent_i64 = i64::from(short_id)
                .checked_sub(initial_diff)
                .expect("parent fits in i64");
            let mut current_parent = u32::try_from(parent_i64).expect("parent fits in u32");
            output.push(current_parent);

            while cursor < end_cursor {
                let gap = read_varint(payload, &mut cursor);
                current_parent = current_parent
                    .checked_sub(gap)
                    .expect("valid descending gap");
                output.push(current_parent);
            }
        }
    }
}

// ---------------------------------------------------------------- rezzy bitmap helpers
fn rezzy_bitmap_sets_bytes(edge_lists: &[Vec<u32>]) -> (usize, usize) {
    let mut total_with_headers = 0usize;
    let mut total_raw_payload = 0usize;
    for target_list in edge_lists {
        if target_list.is_empty() {
            continue;
        }
        let set = BitmapSet::from_values(BENCH_DOMAIN, target_list.iter().copied());
        let encoded_bytes = set.encode().expect("benchmark bitmap must encode");
        total_with_headers = total_with_headers
            .checked_add(encoded_bytes.len())
            .expect("byte count fits");
        // BitmapSet's private wire header is magic (4), version (1), domain (16).
        let header_len = 5usize
            .checked_add(BENCH_DOMAIN.as_bytes().len())
            .expect("header length fits");
        total_raw_payload = total_raw_payload
            .checked_add(
                encoded_bytes
                    .len()
                    .checked_sub(header_len)
                    .expect("bitmap header present"),
            )
            .expect("byte count fits");
    }
    (total_with_headers, total_raw_payload)
}

// ---------------------------------------------------------------- reverse
fn invert_edges(edge_lists: &[Vec<u32>], node_count: usize) -> Vec<Vec<u32>> {
    let mut reversed = vec![Vec::new(); node_count];
    for (child_index, target_list) in edge_lists.iter().enumerate() {
        let child_short = u32::try_from(child_index).expect("fits in u32");
        for &parent_short in target_list {
            let parent_index = usize::try_from(parent_short).expect("fits in usize");
            reversed[parent_index].push(child_short);
        }
    }
    reversed
}

struct CsrGapStorage {
    chunks: Vec<(Vec<u32>, Vec<u8>)>,
}

/// Encode one reverse row: the first child is a zigzag delta from the parent,
/// and later children are ascending gaps. A referenced parent can have a
/// higher short than one of its children, so the first delta is signed.
fn append_reverse_row(parent_short: u32, child_list: &[u32], payload: &mut Vec<u8>) {
    let Some((&first_child, rest)) = child_list.split_first() else {
        return;
    };
    let first_delta = i64::from(first_child)
        .checked_sub(i64::from(parent_short))
        .expect("u32 difference fits i64");
    append_varint_u64(payload, zigzag_encode(first_delta));
    let mut last_child = first_child;
    for &child in rest {
        let gap = child
            .checked_sub(last_child)
            .expect("reverse edge children are stored in ascending order");
        append_varint(payload, gap);
        last_child = child;
    }
}

fn decode_reverse_row(parent_short: u32, payload: &[u8], child_count: usize) -> Vec<u32> {
    let mut children = Vec::with_capacity(child_count);
    if child_count == 0 {
        return children;
    }
    let mut cursor = 0usize;
    let first_delta = zigzag_decode(read_varint_u64(payload, &mut cursor));
    let first = i64::from(parent_short)
        .checked_add(first_delta)
        .and_then(|value| u32::try_from(value).ok())
        .expect("decoded first child fits in u32");
    children.push(first);
    for _ in 1..child_count {
        let gap = read_varint(payload, &mut cursor);
        let next = children
            .last()
            .copied()
            .expect("row is non-empty")
            .checked_add(gap)
            .expect("decoded child does not overflow");
        children.push(next);
    }
    children
}

fn verify_reverse_codec() {
    for (parent, children) in [
        (1_000u32, vec![900u32, 1_001, 1_100]),
        (5u32, vec![6, 7, 9]),
        (10u32, vec![10]),
    ] {
        let mut payload = Vec::new();
        append_reverse_row(parent, &children, &mut payload);
        let decoded = decode_reverse_row(parent, &payload, children.len());
        assert_eq!(decoded, children, "parent short {parent}");
    }
}

struct AdaptiveReverseRezzy {
    csr_storage: CsrGapStorage,
    hub_bitmaps: HashMap<u32, BitmapSet>,
}

impl AdaptiveReverseRezzy {
    fn build(reversed_lists: &[Vec<u32>], hub_threshold: usize) -> Self {
        let mut hub_bitmaps = HashMap::new();
        let mut chunks = Vec::new();

        for (chunk_index, chunk_slice) in reversed_lists.chunks(CHUNK_SIZE).enumerate() {
            let mut offsets = Vec::with_capacity(chunk_slice.len().saturating_add(1));
            let mut payload = Vec::new();

            for (offset_in_chunk, child_list) in chunk_slice.iter().enumerate() {
                offsets.push(u32::try_from(payload.len()).expect("fits in u32"));
                let parent_short = u32::try_from(
                    chunk_index
                        .saturating_mul(CHUNK_SIZE)
                        .saturating_add(offset_in_chunk),
                )
                .expect("short fits in u32");

                if child_list.len() > hub_threshold {
                    let set = BitmapSet::from_values(BENCH_DOMAIN, child_list.iter().copied());
                    hub_bitmaps.insert(parent_short, set);
                    continue;
                }

                append_reverse_row(parent_short, child_list, &mut payload);
            }
            offsets.push(u32::try_from(payload.len()).expect("fits in u32"));
            chunks.push((offsets, payload));
        }

        Self {
            csr_storage: CsrGapStorage { chunks },
            hub_bitmaps,
        }
    }

    fn byte_breakdown(&self) -> (usize, usize) {
        let csr_bytes: usize = self
            .csr_storage
            .chunks
            .iter()
            .map(|(offsets, payload)| {
                offsets
                    .len()
                    .saturating_mul(4)
                    .saturating_add(payload.len())
            })
            .sum();

        let hub_bytes: usize = self
            .hub_bitmaps
            .values()
            .map(|bitmap| bitmap.encode().expect("benchmark bitmap must encode").len())
            .sum();

        (csr_bytes, hub_bytes)
    }

    fn children(&self, parent_short: u32) -> Vec<u32> {
        if let Some(bitmap) = self.hub_bitmaps.get(&parent_short) {
            return bitmap.iter().collect();
        }
        let parent = usize::try_from(parent_short).expect("fits in usize");
        let (offsets, payload) = &self.csr_storage.chunks[parent / CHUNK_SIZE];
        let row = parent % CHUNK_SIZE;
        let mut cursor = usize::try_from(offsets[row]).expect("fits in usize");
        let end =
            usize::try_from(offsets[row.checked_add(1).expect("row fits")]).expect("fits in usize");
        let mut children = Vec::new();
        if cursor < end {
            let first = i64::from(parent_short)
                .checked_add(zigzag_decode(read_varint_u64(payload, &mut cursor)))
                .expect("first child fits");
            let mut child = u32::try_from(first).expect("valid child short");
            children.push(child);
            while cursor < end {
                child = child
                    .checked_add(read_varint(payload, &mut cursor))
                    .expect("child fits");
                children.push(child);
            }
        }
        children
    }

    fn count_children(&self, parent_short: u32) -> usize {
        if let Some(bitmap) = self.hub_bitmaps.get(&parent_short) {
            return usize::try_from(bitmap.len()).unwrap_or(0);
        }

        let parent_usize = usize::try_from(parent_short).expect("fits in usize");
        let (offsets, payload) = &self.csr_storage.chunks[parent_usize / CHUNK_SIZE];
        let within_chunk = parent_usize % CHUNK_SIZE;
        let mut cursor = usize::try_from(offsets[within_chunk]).expect("fits in usize");
        let end_cursor =
            usize::try_from(offsets[within_chunk.saturating_add(1)]).expect("fits in usize");

        let mut child_count = 0usize;
        while cursor < end_cursor {
            black_box(read_varint_u64(payload, &mut cursor));
            child_count = child_count.saturating_add(1);
        }
        child_count
    }
}

// ---------------------------------------------------------------- write amplification
fn measure_forward_wa(
    edge_lists: &[Vec<u32>],
    head_capacity: usize,
    batch_size: usize,
) -> (usize, usize) {
    let mut total_written = 0usize;
    let mut final_stored_bytes = 0usize;
    let mut current_head_bytes = 0usize;
    let mut events_in_head = 0usize;

    for (event_index, target_list) in edge_lists.iter().enumerate() {
        let base_short = u32::try_from(event_index).expect("fits in u32");
        let mut entry_size = 2usize; // u16 offset

        if !target_list.is_empty() {
            let mut sorted = target_list.clone();
            sorted.sort_unstable_by(|a, b| b.cmp(a));
            let diff_0 = i64::from(base_short)
                .checked_sub(i64::from(sorted[0]))
                .expect("valid diff");
            entry_size = entry_size.saturating_add(varint_length_u64(zigzag_encode(diff_0)));

            for window in sorted.windows(2) {
                let gap = window[0].checked_sub(window[1]).expect("sorted descending");
                entry_size = entry_size.saturating_add(varint_length(gap));
            }
        }

        current_head_bytes = current_head_bytes.saturating_add(entry_size);
        events_in_head = events_in_head.saturating_add(1);

        let is_batch_flush = event_index.saturating_add(1).is_multiple_of(batch_size);
        if is_batch_flush || events_in_head == head_capacity {
            total_written = total_written
                .saturating_add(current_head_bytes)
                .saturating_add(2);
        }

        if events_in_head == head_capacity {
            final_stored_bytes = final_stored_bytes
                .saturating_add(current_head_bytes)
                .saturating_add(2);
            current_head_bytes = 0;
            events_in_head = 0;
        }
    }

    if events_in_head > 0 {
        final_stored_bytes = final_stored_bytes
            .saturating_add(current_head_bytes)
            .saturating_add(2);
        total_written = total_written
            .saturating_add(current_head_bytes)
            .saturating_add(2);
    }
    (total_written, final_stored_bytes)
}

struct ReverseWaState {
    node_count: usize,
    hub_threshold: usize,
    in_degrees: Vec<u32>,
    last_seen_child: Vec<u32>,
    row_body_bytes: Vec<usize>,
    chunk_body_bytes: Vec<usize>,
    hub_bitmap_sets: HashMap<u32, BitmapSet>,
    pending_children: HashMap<u32, Vec<u32>>,
    touched_chunks: Vec<usize>,
    touched_hubs: Vec<u32>,
    total_written: usize,
}

impl ReverseWaState {
    fn new(node_count: usize, hub_threshold: usize) -> Self {
        Self {
            node_count,
            hub_threshold,
            in_degrees: vec![0; node_count],
            last_seen_child: vec![0; node_count],
            row_body_bytes: vec![0; node_count],
            chunk_body_bytes: vec![0; node_count.div_ceil(CHUNK_SIZE)],
            hub_bitmap_sets: HashMap::new(),
            // At usize::MAX no row can promote, so retaining children is wasteful.
            pending_children: HashMap::new(),
            touched_chunks: Vec::new(),
            touched_hubs: Vec::new(),
            total_written: 0,
        }
    }

    fn add_edge(&mut self, child_short: u32, parent_short: u32) {
        let parent_index = usize::try_from(parent_short).expect("fits in usize");
        self.in_degrees[parent_index] = self.in_degrees[parent_index].saturating_add(1);
        let degree = self.in_degrees[parent_index];
        if usize::try_from(degree).expect("fits in usize") > self.hub_threshold {
            self.add_hub_edge(child_short, parent_short, parent_index, degree);
        } else {
            self.add_csr_edge(child_short, parent_short, parent_index, degree);
        }
    }

    fn add_hub_edge(
        &mut self,
        child_short: u32,
        parent_short: u32,
        parent_index: usize,
        degree: u32,
    ) {
        let promotion_degree =
            u32::try_from(self.hub_threshold.saturating_add(1)).unwrap_or(u32::MAX);
        let bitmap = self
            .hub_bitmap_sets
            .entry(parent_short)
            .or_insert_with(|| BitmapSet::new(BENCH_DOMAIN));
        if degree == promotion_degree {
            let chunk_idx = parent_index / CHUNK_SIZE;
            self.chunk_body_bytes[chunk_idx] =
                self.chunk_body_bytes[chunk_idx].saturating_sub(self.row_body_bytes[parent_index]);
            self.row_body_bytes[parent_index] = 0;
            push_unique(&mut self.touched_chunks, chunk_idx);
            // Promotion replaces the whole CSR row with a bitmap row.
            for previous in self
                .pending_children
                .remove(&parent_short)
                .unwrap_or_default()
            {
                bitmap.insert(previous);
            }
        }
        bitmap.insert(child_short);
        push_unique(&mut self.touched_hubs, parent_short);
    }

    fn add_csr_edge(
        &mut self,
        child_short: u32,
        parent_short: u32,
        parent_index: usize,
        degree: u32,
    ) {
        let last = if degree == 1 {
            parent_short
        } else {
            self.last_seen_child[parent_index]
        };
        let encoded_len = if degree == 1 {
            varint_length_u64(zigzag_encode(
                i64::from(child_short)
                    .checked_sub(i64::from(last))
                    .expect("u32 difference fits i64"),
            ))
        } else {
            varint_length(
                child_short
                    .checked_sub(last)
                    .expect("children sorted ascending"),
            )
        };
        let chunk_idx = parent_index / CHUNK_SIZE;
        self.chunk_body_bytes[chunk_idx] =
            self.chunk_body_bytes[chunk_idx].saturating_add(encoded_len);
        self.row_body_bytes[parent_index] =
            self.row_body_bytes[parent_index].saturating_add(encoded_len);
        self.last_seen_child[parent_index] = child_short;
        if self.hub_threshold < usize::MAX {
            self.pending_children
                .entry(parent_short)
                .or_default()
                .push(child_short);
        }
        push_unique(&mut self.touched_chunks, chunk_idx);
    }

    fn flush_batch(&mut self) {
        for &chunk_idx in &self.touched_chunks {
            let rows = self.rows_in_chunk(chunk_idx);
            self.total_written = self
                .total_written
                .saturating_add(self.chunk_body_bytes[chunk_idx])
                .saturating_add(4usize.saturating_mul(rows.saturating_add(1)));
        }
        for hub in &self.touched_hubs {
            self.total_written = self.total_written.saturating_add(
                self.hub_bitmap_sets[hub]
                    .encode()
                    .map_or(0, |bytes| bytes.len()),
            );
        }
        self.touched_chunks.clear();
        self.touched_hubs.clear();
    }

    fn rows_in_chunk(&self, chunk_idx: usize) -> usize {
        self.node_count
            .saturating_sub(chunk_idx.saturating_mul(CHUNK_SIZE))
            .min(CHUNK_SIZE)
    }

    fn finish(mut self) -> (usize, usize) {
        // Include writes from the final, possibly partial batch.
        if !self.touched_chunks.is_empty() || !self.touched_hubs.is_empty() {
            self.flush_batch();
        }
        let csr_bytes: usize = self
            .chunk_body_bytes
            .iter()
            .enumerate()
            .map(|(chunk_idx, &body)| {
                body.saturating_add(
                    4usize.saturating_mul(self.rows_in_chunk(chunk_idx).saturating_add(1)),
                )
            })
            .sum();
        let hub_bytes: usize = self
            .hub_bitmap_sets
            .values()
            .map(|bitmap| bitmap.encode().expect("benchmark bitmap must encode").len())
            .sum();
        (self.total_written, csr_bytes.saturating_add(hub_bytes))
    }
}

fn push_unique<T: PartialEq>(items: &mut Vec<T>, item: T) {
    if !items.contains(&item) {
        items.push(item);
    }
}

fn measure_reverse_wa(
    edge_lists: &[Vec<u32>],
    node_count: usize,
    hub_threshold: usize,
    batch_size: usize,
) -> (usize, usize) {
    let mut state = ReverseWaState::new(node_count, hub_threshold);
    for (child_index, target_list) in edge_lists.iter().enumerate() {
        let child_short = u32::try_from(child_index).expect("fits in u32");
        for &parent_short in target_list {
            state.add_edge(child_short, parent_short);
        }
        if child_index.saturating_add(1).is_multiple_of(batch_size) {
            state.flush_batch();
        }
    }
    state.finish()
}

// ---------------------------------------------------------------- timing helper
fn measure_lookup_nanos<F: FnMut(u32)>(query_sample: &[u32], mut lookup_fn: F) -> f64 {
    let timer = Instant::now();
    for &query_id in query_sample {
        lookup_fn(query_id);
    }
    let elapsed_nanos = timer.elapsed().as_nanos();
    let sample_count = query_sample.len().max(1);
    num(u64::try_from(elapsed_nanos).unwrap_or(u64::MAX)) / len_f64(sample_count)
}

// ---------------------------------------------------------------- runner subroutines
fn run_forward_bench(family_name: &str, edge_lists: &[Vec<u32>], queries: &[u32]) {
    let edge_count: usize = edge_lists.iter().map(Vec::len).sum();
    let nonempty_lists = edge_lists.iter().filter(|list| !list.is_empty()).count();

    let print_row = |codec_name: &str,
                     tuple_count: usize,
                     payload_bytes: usize,
                     lookup_nanos: f64| {
        let total_bytes =
            payload_bytes.saturating_add(tuple_count.saturating_mul(TUPLE_OVERHEAD_BYTES));
        let divisor = len_f64(edge_count.max(1));
        let payload_per_edge = len_f64(payload_bytes) / divisor;
        let total_per_edge = len_f64(total_bytes) / divisor;
        println!(
            "{family_name}\t{codec_name}\t{edge_count}\t{tuple_count}\t{payload_bytes}\t{payload_per_edge:.2}\t{total_per_edge:.2}\t{lookup_nanos:.1}"
        );
    };

    // 1. Point tuple per event
    let point_tuple_payload = edge_count.saturating_mul(4);
    let ns_point_tuple = measure_lookup_nanos(queries, |short_id| {
        let index = usize::try_from(short_id).expect("fits in usize");
        black_box(edge_lists[index].as_slice());
    });
    print_row(
        "tuple_per_event_u32",
        nonempty_lists,
        point_tuple_payload,
        ns_point_tuple,
    );

    // 2. CSR raw u32
    let csr_raw = CsrRawForward::build(edge_lists);
    let ns_csr_raw = measure_lookup_nanos(queries, |short_id| {
        black_box(csr_raw.lookup(short_id));
    });
    print_row(
        "csr_raw_u32_c1024",
        csr_raw.chunks.len(),
        csr_raw.total_bytes(),
        ns_csr_raw,
    );

    // 3. CSR varint delta
    let csr_varint = CsrVarintForward::build(edge_lists);
    let mut decode_buffer = Vec::new();
    let ns_csr_varint = measure_lookup_nanos(queries, |short_id| {
        csr_varint.lookup(short_id, &mut decode_buffer);
        black_box(&decode_buffer);
    });
    print_row(
        "csr_varint_delta_c1024",
        csr_varint.chunks.len(),
        csr_varint.total_bytes(),
        ns_csr_varint,
    );

    // 4. Rezzy BitmapSet per event (with 21-byte header)
    let (rezzy_bytes_hdr, rezzy_bytes_raw) = rezzy_bitmap_sets_bytes(edge_lists);
    print_row(
        "rezzy_bmps_per_event",
        nonempty_lists,
        rezzy_bytes_hdr,
        f64::NAN,
    );
    print_row(
        "rezzy_raw_per_event",
        nonempty_lists,
        rezzy_bytes_raw,
        f64::NAN,
    );
}

fn run_reverse_bench(
    family_name: &str,
    edge_lists: &[Vec<u32>],
    node_count: usize,
    queries: &[u32],
) {
    let reversed = invert_edges(edge_lists, node_count);
    let edge_count: usize = reversed.iter().map(Vec::len).sum();
    let parent_count = reversed.iter().filter(|list| !list.is_empty()).count();
    let valid_queries: Vec<u32> = queries
        .iter()
        .copied()
        .filter(|&parent| {
            let parent_idx = usize::try_from(parent).expect("fits in usize");
            !reversed[parent_idx].is_empty()
        })
        .collect();

    let divisor = len_f64(edge_count.max(1));

    // 1. Point tuple per parent
    let tuple_payload = edge_count.saturating_mul(4);
    let tuple_total =
        tuple_payload.saturating_add(parent_count.saturating_mul(TUPLE_OVERHEAD_BYTES));
    println!(
        "{family_name}\ttuple_per_parent_u32\t{edge_count}\t{parent_count}\t0\t{parent_count}\t{tuple_payload}\t{:.2}\t{:.2}\t-\t-",
        len_f64(tuple_payload) / divisor,
        len_f64(tuple_total) / divisor
    );

    // 2. Rezzy BitmapSet per parent
    let (rezzy_hdr, _) = rezzy_bitmap_sets_bytes(&reversed);
    let rezzy_total = rezzy_hdr.saturating_add(parent_count.saturating_mul(TUPLE_OVERHEAD_BYTES));
    println!(
        "{family_name}\trezzy_per_parent\t{edge_count}\t{parent_count}\t0\t{parent_count}\t{rezzy_hdr}\t{:.2}\t{:.2}\t-\t-",
        len_f64(rezzy_hdr) / divisor,
        len_f64(rezzy_total) / divisor
    );

    // 3. Adaptive reverse with Rezzy BitmapSet for hubs
    for &hub_threshold in &[usize::MAX, 4096, 256] {
        let adaptive = AdaptiveReverseRezzy::build(&reversed, hub_threshold);
        let (csr_bytes, hub_bytes) = adaptive.byte_breakdown();
        let payload_bytes = csr_bytes.saturating_add(hub_bytes);
        let tuple_count = adaptive
            .csr_storage
            .chunks
            .len()
            .saturating_add(adaptive.hub_bitmaps.len());
        let total_bytes =
            payload_bytes.saturating_add(tuple_count.saturating_mul(TUPLE_OVERHEAD_BYTES));

        let ns_scan = measure_lookup_nanos(&valid_queries, |parent_short| {
            black_box(adaptive.count_children(parent_short));
        });

        let ns_hub = if !edge_lists.is_empty() && family_name == "auth" {
            let hub_queries: Vec<u32> = vec![0; 200];
            measure_lookup_nanos(&hub_queries, |parent_short| {
                black_box(adaptive.count_children(parent_short));
            })
        } else {
            f64::NAN
        };

        let label = if hub_threshold == usize::MAX {
            "csr_gap_varint_c1024".to_owned()
        } else {
            format!("rezzy_adaptive_hub>{hub_threshold}")
        };

        println!(
            "{family_name}\t{label}\t{edge_count}\t{parent_count}\t{}\t{tuple_count}\t{payload_bytes}\t{:.2}\t{:.2}\t{ns_scan:.1}\t{ns_hub:.0}",
            adaptive.hub_bitmaps.len(),
            len_f64(payload_bytes) / divisor,
            len_f64(total_bytes) / divisor
        );
    }
}

fn run_write_amplification_bench(room: &RoomGraph, event_count: usize, relations_fwd: &[Vec<u32>]) {
    println!("\n# head/sealed write amplification (bytes written / final bytes)");
    println!("family\tside\thead_or_hub\tbatch\twritten_B\tfinal_B\tWA");

    for (family_name, edge_lists) in [("prev", &room.prev), ("auth", &room.auth)] {
        for &head_cap in &[64usize, 256, 1024] {
            for &batch_sz in &[1usize, 16, 128] {
                let (written, final_sz) = measure_forward_wa(edge_lists, head_cap, batch_sz);
                let wa = len_f64(written) / len_f64(final_sz.max(1));
                println!("{family_name}\tforward\thead={head_cap}\t{batch_sz}\t{written}\t{final_sz}\t{wa:.1}");
            }
        }
    }

    for (family_name, edge_lists) in [
        ("prev", room.prev.as_slice()),
        ("auth", room.auth.as_slice()),
        ("rel", relations_fwd),
    ] {
        for &batch_sz in &[1usize, 16, 128] {
            let (written, final_sz) = measure_reverse_wa(edge_lists, event_count, 4096, batch_sz);
            let wa = len_f64(written) / len_f64(final_sz.max(1));
            println!(
                "{family_name}\treverse\thub>4096\t{batch_sz}\t{written}\t{final_sz}\t{wa:.1}"
            );
        }
    }
}

// ---------------------------------------------------------------- main entry
fn main() {
    verify_reverse_codec();
    if std::env::args().nth(1).as_deref() == Some("--sweep") {
        sensitivity::run();
        return;
    }
    // Round-trip the late-parent case absent from the locality-biased generator.
    // Exercise both sparse arrays and promotion to a hub bitmap.
    let edges = vec![vec![4, 2], vec![4], vec![], vec![], vec![0]];
    let forward = CsrVarintForward::build(&edges);
    for (short, expected) in edges.iter().enumerate() {
        let mut decoded = Vec::new();
        forward.lookup(u32::try_from(short).expect("short fits"), &mut decoded);
        decoded.sort_unstable();
        let mut expected = expected.clone();
        expected.sort_unstable();
        assert_eq!(decoded, expected, "forward late-parent row {short}");
    }
    let reversed = invert_edges(&edges, edges.len());
    for threshold in [0, usize::MAX] {
        let reverse = AdaptiveReverseRezzy::build(&reversed, threshold);
        for (parent, expected) in reversed.iter().enumerate() {
            let parent = u32::try_from(parent).expect("short fits");
            assert_eq!(
                reverse.children(parent),
                *expected,
                "reverse late-parent row {parent}, threshold {threshold}"
            );
            assert_eq!(
                reverse.count_children(parent),
                expected.len(),
                "reverse count for {parent}"
            );
        }
    }

    let args: Vec<String> = std::env::args().collect();
    let event_count: usize = args
        .get(1)
        .and_then(|arg_str| arg_str.parse().ok())
        .unwrap_or(1_000_000);
    assert!(event_count >= 3, "event_count must be at least 3");
    let random_seed: u64 = args
        .get(2)
        .and_then(|arg_str| arg_str.parse().ok())
        .unwrap_or(0x5EED);

    let room = generate_room(event_count, random_seed);
    let relations_forward: Vec<Vec<u32>> = room
        .relations
        .iter()
        .map(|rel_opt| rel_opt.map(|(parent, _)| vec![parent]).unwrap_or_default())
        .collect();

    let mut query_rng = PseudoRng(random_seed ^ 0xABCD);
    let max_query_bound = u64::try_from(event_count)
        .expect("fits in u64")
        .saturating_sub(2);
    let query_sample: Vec<u32> = (0..200_000)
        .map(|_| {
            let offset = u32::try_from(query_rng.below(max_query_bound)).expect("fits in u32");
            2u32.saturating_add(offset)
        })
        .collect();

    println!("events\t{event_count}");
    println!("\n# forward\nfamily\tencoding\tedges\ttuples\tpayload_B\tB_per_edge\tB_per_edge_incl_tuples\tlookup_ns");
    for (family_name, edge_lists) in [
        ("prev", &room.prev),
        ("auth", &room.auth),
        ("rel", &relations_forward),
    ] {
        run_forward_bench(family_name, edge_lists, &query_sample);
    }

    println!("\n# reverse (children-of)\nfamily\tencoding\tedges\tparents\thubs\ttuples\tpayload_B\tB_per_edge\tB_per_edge_incl_tuples\tcount_ns\thub_count_ns");
    for (family_name, edge_lists) in [
        ("prev", &room.prev),
        ("auth", &room.auth),
        ("rel", &relations_forward),
    ] {
        run_reverse_bench(family_name, edge_lists, event_count, &query_sample);
    }

    run_write_amplification_bench(&room, event_count, &relations_forward);
}
