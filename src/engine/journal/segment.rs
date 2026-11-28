//! On-disk segment framing: file header, mark slots, and commit groups.
//!
//! These are the pure codecs over the journal byte layout. The [`Journal`] and
//! [`JournalCoordinator`] state machines live in the parent module.

#[cfg(test)]
use super::{fs, OpenOptions};
use super::{
    io, pool_from_tag, pool_tag, CommittedGroup, File, Hasher, JournalEntry, Mutation, Path, Read,
    ReclaimedHistory, Scan, Seek, SeekFrom, ShardType, Write, CONSUMED_TAIL_LEN, FH_BASE_LSN,
    FH_BASE_SEQUENCE, FH_MAGIC, FH_MARKS_AT, FH_SLOTS, FH_VERSION, FILE_HEADER_LEN, FILE_MAGIC,
    FRAME_FIXED_LEN, FRAME_TRAILER_LEN, GH_CRC, GH_CRC_COVERED, GH_FIRST_LSN, GH_HEADER_LEN,
    GH_LAST_LSN, GH_MAGIC, GH_PAYLOAD_LEN, GH_RECORD_COUNT, GH_SEQUENCE, GROUP_COMMIT_MAGIC,
    GROUP_HEADER_LEN, GROUP_MAGIC, GROUP_TRAILER_LEN, GT_CRC, GT_MAGIC, GT_SEQUENCE,
    JOURNAL_VERSION, MARK_SLOT_LEN, MARK_SLOT_OFFSETS, MAX_GROUP_LEN, MF_COLLECTION, MF_FLAGS,
    MF_KIND, MF_LSN, MF_NODE, MF_PAYLOAD_LEN, MF_POOL, MIN_FRAME_LEN, POOL_SLOTS,
};

/// Append `len` bytes of the file at `path`, starting at `start`, to `out`.
pub(super) fn read_range_into(
    path: &Path,
    start: u64,
    len: u64,
    out: &mut Vec<u8>,
) -> io::Result<()> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(start))?;
    let len = usize::try_from(len).map_err(|_| invalid_data("range too large"))?;
    let filled = out.len();
    out.resize(filled.saturating_add(len), 0);
    if let Err(error) = file.read_exact(out.get_mut(filled..).unwrap_or_default()) {
        out.truncate(filled);
        return Err(error);
    }
    Ok(())
}

/// The decoded file header.
pub(super) struct FileHeader {
    pub(super) base_sequence: u64,
    pub(super) base_lsn: u64,
    /// Per pool, the newest reclaimed-history LSN recorded in this header.
    pub(super) reclaimed: ReclaimedHistory,
}

pub(super) fn file_header_bytes(
    base_sequence: u64,
    base_lsn: u64,
    reclaimed: ReclaimedHistory,
) -> Vec<u8> {
    let mut header = vec![0_u8; FILE_HEADER_LEN];
    header[FH_MAGIC].copy_from_slice(FILE_MAGIC);
    header[FH_VERSION].copy_from_slice(&JOURNAL_VERSION.to_le_bytes());
    header[FH_BASE_SEQUENCE].copy_from_slice(&base_sequence.to_le_bytes());
    header[FH_BASE_LSN].copy_from_slice(&base_lsn.to_le_bytes());
    let slots = u32::try_from(POOL_SLOTS).unwrap_or(u32::MAX);
    header[FH_SLOTS].copy_from_slice(&slots.to_le_bytes());
    let mut end = FH_MARKS_AT;
    for mark in reclaimed {
        let next = end.saturating_add(8);
        header[end..next].copy_from_slice(&mark.to_le_bytes());
        end = next;
    }
    let mut crc = Hasher::new();
    crc.update(&header[..end]);
    let crc_end = end.saturating_add(4);
    header[end..crc_end].copy_from_slice(&crc.finalize().to_le_bytes());
    header
}

pub(super) fn write_file_header(
    file: &mut File,
    base_sequence: u64,
    base_lsn: u64,
    reclaimed: ReclaimedHistory,
) -> io::Result<()> {
    file.write_all(&file_header_bytes(base_sequence, base_lsn, reclaimed))
}

/// Validate the file header and decode it. A rotated segment records the first
/// surviving group's numbers here so scanning never has to assume the sequence
/// begins at 1. The header always names one reclaimed-history slot per pool;
/// any fault, including a bad checksum over the marks, is an error.
pub(super) fn read_file_header(bytes: &[u8]) -> io::Result<FileHeader> {
    let Some(header) = bytes.get(..FILE_HEADER_LEN) else {
        return Err(invalid_data("truncated journal file header"));
    };
    if &header[FH_MAGIC] != FILE_MAGIC {
        return Err(invalid_data("invalid journal magic"));
    }
    let version = u32::from_le_bytes(header[FH_VERSION].try_into().expect("fixed slice"));
    if version != JOURNAL_VERSION {
        return Err(invalid_data("unsupported journal version"));
    }
    let base_sequence =
        u64::from_le_bytes(header[FH_BASE_SEQUENCE].try_into().expect("fixed slice"));
    let base_lsn = u64::from_le_bytes(header[FH_BASE_LSN].try_into().expect("fixed slice"));
    let slots = u32::from_le_bytes(header[FH_SLOTS].try_into().expect("fixed slice"));
    if usize::try_from(slots).unwrap_or(usize::MAX) != POOL_SLOTS {
        return Err(invalid_data("journal header names an unknown pool count"));
    }
    let end = FH_MARKS_AT.saturating_add(POOL_SLOTS.saturating_mul(8));
    let crc_end = end.saturating_add(4);
    let Some(stored) = header.get(end..crc_end) else {
        return Err(invalid_data("truncated journal file header"));
    };
    let mut crc = Hasher::new();
    crc.update(&header[..end]);
    if u32::from_le_bytes(stored.try_into().expect("fixed slice")) != crc.finalize() {
        return Err(invalid_data("journal header checksum mismatch"));
    }
    let mut reclaimed = [0_u64; POOL_SLOTS];
    for (index, mark) in reclaimed.iter_mut().enumerate() {
        let at = FH_MARKS_AT.saturating_add(index.saturating_mul(8));
        let next = at.saturating_add(8);
        *mark = u64::from_le_bytes(header[at..next].try_into().expect("fixed slice"));
    }
    Ok(FileHeader {
        base_sequence,
        base_lsn,
        reclaimed,
    })
}

/// [`read_file_header`], for the callers that need only the base numbers.
pub(super) fn validate_file_header(bytes: &[u8]) -> io::Result<(u64, u64)> {
    let header = read_file_header(bytes)?;
    Ok((header.base_sequence, header.base_lsn))
}

/// One durability-mark slot: which generation it is, and what it claims.
#[derive(Clone, Copy)]
pub(super) struct MarkSlot {
    pub(super) generation: u64,
    pub(super) durable_len: u64,
}

pub(super) fn encode_mark_slot(slot: MarkSlot) -> [u8; MARK_SLOT_LEN] {
    let mut bytes = [0_u8; MARK_SLOT_LEN];
    bytes[..8].copy_from_slice(&slot.generation.to_le_bytes());
    bytes[8..16].copy_from_slice(&slot.durable_len.to_le_bytes());
    let mut crc = Hasher::new();
    crc.update(&bytes[..16]);
    bytes[16..20].copy_from_slice(&crc.finalize().to_le_bytes());
    bytes
}

/// Decode one slot: `None` if it is unwritten (all zero) or fails its checksum.
pub(super) fn decode_mark_slot(bytes: &[u8]) -> Option<MarkSlot> {
    let bytes = bytes.get(..MARK_SLOT_LEN)?;
    let mut crc = Hasher::new();
    crc.update(&bytes[..16]);
    if u32::from_le_bytes(bytes[16..20].try_into().ok()?) != crc.finalize() {
        return None;
    }
    let slot = MarkSlot {
        generation: u64::from_le_bytes(bytes[..8].try_into().ok()?),
        durable_len: u64::from_le_bytes(bytes[8..16].try_into().ok()?),
    };
    // Generation zero is never written, so a checksum-valid slot claiming it
    // is not a mark.
    (slot.generation != 0).then_some(slot)
}

/// The newest valid mark in a segment's header region, or `None` when neither
/// slot holds one: unwritten, torn, or corrupt. `None` means "durability
/// unknown", which only ever widens what recovery is willing to truncate.
pub(super) fn newest_mark(bytes: &[u8]) -> Option<MarkSlot> {
    MARK_SLOT_OFFSETS
        .iter()
        .filter_map(|&offset| bytes.get(offset..).and_then(decode_mark_slot))
        .max_by_key(|slot| slot.generation)
}

/// Test-only: cut a **copy** of a journal segment back to its embedded
/// durability mark, keeping the header. That is the worst case of a power cut:
/// every frame written after the last fsync is lost.
///
/// # Errors
/// Returns an error if the copy cannot be read, has an invalid header, or cannot
/// be truncated.
#[cfg(test)]
pub(crate) fn cut_segment_image_to_durable_mark(path: &Path) -> io::Result<()> {
    let bytes = fs::read(path)?;
    if bytes.len() < FILE_HEADER_LEN {
        return Ok(());
    }
    let (_, _) = validate_file_header(&bytes)?;
    let file_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    let mark = read_durable_len_from_header(&bytes, file_len);
    let header = u64::try_from(FILE_HEADER_LEN).unwrap_or(u64::MAX);
    let keep = mark.max(header).min(file_len);
    OpenOptions::new().write(true).open(path)?.set_len(keep)
}

/// How many leading bytes of the segment are known durable. A missing or unreadable mark, or one claiming more than the file holds,
/// yields `0`.
pub(super) fn read_durable_len_from_header(bytes: &[u8], file_len: u64) -> u64 {
    newest_mark(bytes)
        .map(|slot| slot.durable_len)
        .filter(|&durable_len| durable_len <= file_len)
        .unwrap_or(0)
}

pub(super) fn encode_group_header(
    sequence: u64,
    first_lsn: u64,
    last_lsn: u64,
    payload_len: u64,
    record_count: u32,
) -> [u8; GROUP_HEADER_LEN] {
    let mut header = [0_u8; GROUP_HEADER_LEN];
    let header_len = u32::try_from(GROUP_HEADER_LEN).expect("GROUP_HEADER_LEN must fit in a u32");
    header[GH_MAGIC].copy_from_slice(GROUP_MAGIC);
    header[GH_HEADER_LEN].copy_from_slice(&header_len.to_le_bytes());
    header[GH_SEQUENCE].copy_from_slice(&sequence.to_le_bytes());
    header[GH_FIRST_LSN].copy_from_slice(&first_lsn.to_le_bytes());
    header[GH_LAST_LSN].copy_from_slice(&last_lsn.to_le_bytes());
    header[GH_PAYLOAD_LEN].copy_from_slice(&payload_len.to_le_bytes());
    header[GH_RECORD_COUNT].copy_from_slice(&record_count.to_le_bytes());
    let mut crc = Hasher::new();
    crc.update(&header[GH_CRC_COVERED]);
    header[GH_CRC].copy_from_slice(&crc.finalize().to_le_bytes());
    header
}

pub(super) fn encode_group_trailer(sequence: u64, group_crc: u32) -> [u8; GROUP_TRAILER_LEN] {
    let mut trailer = [0_u8; GROUP_TRAILER_LEN];
    trailer[GT_MAGIC].copy_from_slice(GROUP_COMMIT_MAGIC);
    trailer[GT_SEQUENCE].copy_from_slice(&sequence.to_le_bytes());
    trailer[GT_CRC].copy_from_slice(&group_crc.to_le_bytes());
    trailer
}

/// Re-encode a committed group with its original sequence and LSNs, for a
/// segment rewrite. Mirrors exactly the framing `commit_group` writes.
pub(super) fn encode_group(group: &CommittedGroup, into: &mut Vec<u8>) -> io::Result<()> {
    let mut payload = Vec::new();
    for entry in &group.entries {
        encode_mutation(entry.lsn, entry.pool, &entry.mutation, &mut payload)?;
    }
    let record_count = u32::try_from(group.entries.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "too many journal mutations"))?;
    let payload_len = u64::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "journal group too large"))?;
    let header = encode_group_header(
        group.sequence,
        group.first_lsn,
        group.last_lsn,
        payload_len,
        record_count,
    );
    let mut group_crc = Hasher::new();
    group_crc.update(&header);
    group_crc.update(&payload);
    let trailer = encode_group_trailer(group.sequence, group_crc.finalize());
    into.extend_from_slice(&header);
    into.extend_from_slice(&payload);
    into.extend_from_slice(&trailer);
    Ok(())
}

pub(super) fn encode_mutation(
    lsn: u64,
    pool: ShardType,
    mutation: &Mutation,
    into: &mut Vec<u8>,
) -> io::Result<()> {
    // The flag byte is always the pool tag.
    let pool_byte = pool_tag(pool);

    let start = into.len();
    // Zeroed fixed area; `MF_FLAGS` and the DeleteCollection-only fields stay
    // zero by construction. Index the fixed area by its named field ranges so
    // no raw offset arithmetic is needed.
    into.resize(start.saturating_add(FRAME_FIXED_LEN), 0);
    let payload = {
        let frame = &mut into[start..];
        frame[MF_KIND].copy_from_slice(&match mutation {
            Mutation::Put { .. } => [1],
            Mutation::DeleteCollection { .. } => [2],
            Mutation::RemoveRecord { .. } => [3],
        });
        frame[MF_POOL].copy_from_slice(&[pool_byte]);
        frame[MF_LSN].copy_from_slice(&lsn.to_le_bytes());
        match mutation {
            Mutation::Put {
                collection_id,
                node_id,
                payload,
            } => {
                frame[MF_COLLECTION].copy_from_slice(collection_id);
                frame[MF_NODE].copy_from_slice(node_id);
                let payload_len = u32::try_from(payload.len()).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "mutation payload exceeds u32")
                })?;
                frame[MF_PAYLOAD_LEN].copy_from_slice(&payload_len.to_le_bytes());
                payload.as_slice()
            }
            Mutation::DeleteCollection { collection_id } => {
                frame[MF_COLLECTION].copy_from_slice(collection_id);
                &[]
            }
            Mutation::RemoveRecord {
                collection_id,
                node_id,
            } => {
                frame[MF_COLLECTION].copy_from_slice(collection_id);
                frame[MF_NODE].copy_from_slice(node_id);
                &[]
            }
        }
    };
    into.extend_from_slice(payload);
    let mut crc = Hasher::new();
    crc.update(&into[start..]);
    into.extend_from_slice(&crc.finalize().to_le_bytes());
    Ok(())
}

/// Validated fixed fields of one committed group header.
pub(super) struct GroupHeader {
    sequence: u64,
    first_lsn: u64,
    last_lsn: u64,
    payload_len: u64,
    record_count: u32,
}

/// Validate a group header's magic, length, and CRC, returning its fields.
pub(super) fn parse_group_header(header: &[u8]) -> io::Result<GroupHeader> {
    if &header[..4] != GROUP_MAGIC
        || u32::from_le_bytes(header[4..8].try_into().expect("fixed slice"))
            != u32::try_from(GROUP_HEADER_LEN).expect("GROUP_HEADER_LEN fits in u32")
    {
        return Err(invalid_data("invalid journal group header"));
    }
    let mut header_crc = Hasher::new();
    header_crc.update(&header[..44]);
    if u32::from_le_bytes(header[44..48].try_into().expect("fixed slice")) != header_crc.finalize()
    {
        return Err(invalid_data("journal group header checksum mismatch"));
    }
    Ok(GroupHeader {
        sequence: u64::from_le_bytes(header[8..16].try_into().expect("fixed slice")),
        first_lsn: u64::from_le_bytes(header[16..24].try_into().expect("fixed slice")),
        last_lsn: u64::from_le_bytes(header[24..32].try_into().expect("fixed slice")),
        payload_len: u64::from_le_bytes(header[32..40].try_into().expect("fixed slice")),
        record_count: u32::from_le_bytes(header[40..44].try_into().expect("fixed slice")),
    })
}

/// Verify a group's commit trailer and whole-group CRC, returning its payload.
pub(super) fn verify_group_payload<'a>(
    bytes: &'a [u8],
    header: &[u8],
    group: &GroupHeader,
    payload_start: usize,
    payload_len: usize,
) -> io::Result<&'a [u8]> {
    let payload_end = payload_start
        .checked_add(payload_len)
        .ok_or_else(|| invalid_data("journal group length overflow"))?;
    let trailer_end = payload_end
        .checked_add(GROUP_TRAILER_LEN)
        .ok_or_else(|| invalid_data("journal group length overflow"))?;
    let trailer = bytes
        .get(payload_end..trailer_end)
        .ok_or_else(|| invalid_data("truncated journal commit trailer"))?;
    if &trailer[..4] != GROUP_COMMIT_MAGIC
        || u64::from_le_bytes(trailer[4..12].try_into().expect("fixed slice")) != group.sequence
    {
        return Err(invalid_data("invalid journal commit trailer"));
    }
    let payload = bytes
        .get(payload_start..payload_end)
        .ok_or_else(|| invalid_data("truncated journal group payload"))?;
    let mut group_crc = Hasher::new();
    group_crc.update(header);
    group_crc.update(payload);
    if u32::from_le_bytes(trailer[12..16].try_into().expect("fixed slice")) != group_crc.finalize()
    {
        return Err(invalid_data("journal committed group checksum mismatch"));
    }
    Ok(payload)
}

pub(super) fn scan_bytes(
    bytes: &[u8],
    base_sequence: u64,
    base_lsn: u64,
    durable_len: impl Fn() -> u64,
) -> io::Result<Scan> {
    if bytes.len() < FILE_HEADER_LEN {
        return Err(invalid_data("truncated journal file header"));
    }
    scan_groups_from(
        &bytes[FILE_HEADER_LEN..],
        u64::try_from(FILE_HEADER_LEN).unwrap_or(u64::MAX),
        base_sequence,
        base_lsn,
        base_lsn,
        base_sequence,
        durable_len,
    )
}

/// What validating one group at a cursor found.
pub(super) enum GroupStep {
    /// A complete, valid group that occupies `total_len` bytes.
    Group {
        group: GroupHeader,
        entries: Vec<JournalEntry>,
        total_len: usize,
    },
    /// Not enough bytes remain for a complete group.
    Incomplete,
}

/// Why a group failed validation, split by whether a crash can explain it.
pub(super) enum GroupFault {
    /// Damage that a torn or missing page can produce: a bad group header or a
    /// bad commit trailer or checksum. Above the durable mark this is an
    /// unacknowledged crash tail; at or below it, it is corruption.
    Crash(io::Error),
    /// A group whose header checksum verified but whose contents are wrong
    /// (sequence or LSN regression, impossible bounds, undecodable records).
    /// A crash cannot do that to an append-only file, so this is a bug or real
    /// corruption wherever it appears and is never truncated.
    Fatal(io::Error),
}

/// Validate the group starting at `cursor` in `bytes`.
pub(super) fn scan_one_group(
    bytes: &[u8],
    cursor: usize,
    base_offset: u64,
    expected_sequence: u64,
    expected_lsn: u64,
) -> Result<GroupStep, GroupFault> {
    let remaining = bytes.len().saturating_sub(cursor);
    if remaining < GROUP_HEADER_LEN {
        return Ok(GroupStep::Incomplete);
    }
    let header = bytes
        .get(cursor..cursor.saturating_add(GROUP_HEADER_LEN))
        .ok_or_else(|| GroupFault::Crash(invalid_data("truncated journal group header")))?;
    let group = parse_group_header(header).map_err(GroupFault::Crash)?;
    if group.sequence < expected_sequence
        || group.first_lsn != expected_lsn
        || group.last_lsn < group.first_lsn
        || group
            .last_lsn
            .saturating_sub(group.first_lsn)
            .saturating_add(1)
            != u64::from(group.record_count)
        || group.record_count == 0
        || group.payload_len > MAX_GROUP_LEN
        || u64::from(group.record_count).saturating_mul(MIN_FRAME_LEN as u64) > group.payload_len
    {
        return Err(GroupFault::Fatal(invalid_data(
            "invalid journal sequence or group bounds",
        )));
    }
    let payload_len = usize::try_from(group.payload_len).map_err(|_| {
        GroupFault::Fatal(invalid_data("journal group length exceeds address space"))
    })?;
    let total_len = GROUP_HEADER_LEN
        .checked_add(payload_len)
        .and_then(|len| len.checked_add(GROUP_TRAILER_LEN))
        .ok_or_else(|| GroupFault::Fatal(invalid_data("journal group length overflow")))?;
    if remaining < total_len {
        return Ok(GroupStep::Incomplete);
    }
    let payload_index = cursor.saturating_add(GROUP_HEADER_LEN);
    let payload = verify_group_payload(bytes, header, &group, payload_index, payload_len)
        .map_err(GroupFault::Crash)?;
    let payload_offset = base_offset
        .saturating_add(u64::try_from(payload_index).unwrap_or(u64::MAX))
        .try_into()
        .unwrap_or(usize::MAX);
    let entries = decode_mutations(payload, group.first_lsn, group.record_count, payload_offset)
        .map_err(GroupFault::Fatal)?;
    Ok(GroupStep::Group {
        group,
        entries,
        total_len,
    })
}

/// Scan complete groups from `bytes`, which begins at absolute file offset
/// `base_offset`.
///
/// `expected_sequence` is a floor, not an exact match: a segment sharing a
/// global sequence with other pools may skip values. `expected_lsn` must match
/// the first group's `first_lsn` exactly, since LSNs stay contiguous within a
/// segment. `Scan::valid_len` is absolute in the file, not relative to `bytes`.
///
/// `durable_len` yields the embedded durability mark: how many
/// leading bytes of the file a completed fsync covered. It is called at most
/// once, and only when a group fails validation in a way a crash can explain. A group that fails validation
/// in a way a crash can explain ([`GroupFault::Crash`]) at or above it was never
/// acknowledged, so the scan stops there and reports a truncated tail, dropping
/// everything after it, even later groups that happen to be intact. The same
/// failure below it is corruption and is returned as an error. Return `u64::MAX`
/// to treat every byte as durable, so any invalid group fails; return `0` when
/// nothing is known, so the first invalid group ends the scan.
pub(super) fn scan_groups_from(
    bytes: &[u8],
    base_offset: u64,
    mut expected_sequence: u64,
    mut expected_lsn: u64,
    segment_base_lsn: u64,
    segment_base_sequence: u64,
    durable_len: impl Fn() -> u64,
) -> io::Result<Scan> {
    let mut cursor = 0usize;
    let mut valid_len = base_offset;
    let mut groups = Vec::new();
    let mut group_ends = Vec::new();
    let mut truncated_tail = false;

    while cursor < bytes.len() {
        let (group, entries, total_len) =
            match scan_one_group(bytes, cursor, base_offset, expected_sequence, expected_lsn) {
                Ok(GroupStep::Group {
                    group,
                    entries,
                    total_len,
                }) => (group, entries, total_len),
                Ok(GroupStep::Incomplete) => {
                    truncated_tail = true;
                    break;
                }
                Err(GroupFault::Fatal(error)) => return Err(error),
                Err(GroupFault::Crash(error)) => {
                    let at = base_offset.saturating_add(u64::try_from(cursor).unwrap_or(u64::MAX));
                    // The durability mark matters only here, on a failure a crash
                    // can explain, so it is read only now. Reading it on every scan
                    // would cost a file open and read on the hot incremental path
                    // for something a healthy segment never consults.
                    if at >= durable_len() {
                        truncated_tail = true;
                        break;
                    }
                    return Err(error);
                }
            };
        groups.push(CommittedGroup {
            sequence: group.sequence,
            first_lsn: group.first_lsn,
            last_lsn: group.last_lsn,
            entries,
        });
        cursor = cursor.saturating_add(total_len);
        valid_len = base_offset.saturating_add(u64::try_from(cursor).unwrap_or(u64::MAX));
        group_ends.push(valid_len);
        expected_sequence = group
            .sequence
            .checked_add(1)
            .ok_or_else(|| invalid_data("journal group sequence overflow"))?;
        expected_lsn = group
            .last_lsn
            .checked_add(1)
            .ok_or_else(|| invalid_data("journal LSN overflow"))?;
    }

    let consumed_tail = bytes[cursor.saturating_sub(CONSUMED_TAIL_LEN)..cursor].to_vec();
    Ok(Scan {
        groups,
        valid_len,
        truncated_tail,
        base_lsn: segment_base_lsn,
        base_sequence: segment_base_sequence,
        consumed_tail,
        group_ends,
    })
}

pub(super) fn decode_mutations(
    payload: &[u8],
    first_lsn: u64,
    record_count: u32,
    payload_offset: usize,
) -> io::Result<Vec<JournalEntry>> {
    let mut cursor = 0_usize;
    let mut entries = Vec::with_capacity(usize::try_from(record_count).unwrap_or(0));
    for index in 0..record_count {
        let start = cursor;
        if payload.len().saturating_sub(cursor) < MIN_FRAME_LEN {
            return Err(invalid_data("truncated journal mutation frame"));
        }
        let fixed_end = cursor
            .checked_add(FRAME_FIXED_LEN)
            .ok_or_else(|| invalid_data("journal mutation length overflow"))?;
        let frame = payload
            .get(cursor..fixed_end)
            .ok_or_else(|| invalid_data("truncated journal mutation frame"))?;
        let kind = frame[MF_KIND.start];
        let tag = frame[MF_POOL.start];
        if frame[MF_FLAGS] != [0, 0] {
            return Err(invalid_data("unsupported journal mutation flags"));
        }
        let pool = pool_from_tag(tag)
            .ok_or_else(|| invalid_data("journal frame has no valid pool tag"))?;
        let lsn = u64::from_le_bytes(frame[MF_LSN].try_into().expect("fixed slice"));
        let expected = first_lsn
            .checked_add(u64::from(index))
            .ok_or_else(|| invalid_data("journal mutation LSN overflow"))?;
        if lsn != expected {
            return Err(invalid_data("journal mutation LSN out of order"));
        }
        let collection_id: [u8; 16] = frame[MF_COLLECTION].try_into().expect("fixed slice");
        let node_id: [u8; 16] = frame[MF_NODE].try_into().expect("fixed slice");
        let payload_len = usize::try_from(u32::from_le_bytes(
            frame[MF_PAYLOAD_LEN].try_into().expect("fixed slice"),
        ))
        .map_err(|_| invalid_data("journal mutation payload length exceeds address space"))?;
        let frame_end = fixed_end
            .checked_add(payload_len)
            .ok_or_else(|| invalid_data("journal mutation length overflow"))?;
        let crc_end = frame_end
            .checked_add(FRAME_TRAILER_LEN)
            .ok_or_else(|| invalid_data("journal mutation length overflow"))?;
        let data = payload
            .get(fixed_end..frame_end)
            .ok_or_else(|| invalid_data("journal mutation payload exceeds group"))?;
        let crc_bytes = payload
            .get(frame_end..crc_end)
            .ok_or_else(|| invalid_data("journal mutation checksum exceeds group"))?;
        let mut crc = Hasher::new();
        crc.update(
            payload
                .get(start..frame_end)
                .ok_or_else(|| invalid_data("journal mutation frame exceeds group"))?,
        );
        if u32::from_le_bytes(crc_bytes.try_into().expect("fixed slice")) != crc.finalize() {
            return Err(invalid_data("journal mutation checksum mismatch"));
        }
        let mutation = match kind {
            1 => Mutation::Put {
                collection_id,
                node_id,
                payload: data.to_vec(),
            },
            2 if payload_len == 0 && node_id == [0; 16] => {
                Mutation::DeleteCollection { collection_id }
            }
            3 if payload_len == 0 && node_id != [0; 16] => Mutation::RemoveRecord {
                collection_id,
                node_id,
            },
            _ => return Err(invalid_data("unknown or malformed journal mutation kind")),
        };
        entries.push(JournalEntry {
            lsn,
            offset: u64::try_from(payload_offset.saturating_add(start)).unwrap_or(u64::MAX),
            frame_len: u64::try_from(crc_end.saturating_sub(start)).unwrap_or(u64::MAX),
            pool,
            mutation,
        });
        cursor = crc_end;
    }
    if cursor != payload.len() {
        return Err(invalid_data("extra bytes after journal mutation frames"));
    }
    Ok(entries)
}

pub(super) fn sync_parent_dir(path: &Path) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    crate::shard::sync_directory(parent)
}

pub(super) fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
