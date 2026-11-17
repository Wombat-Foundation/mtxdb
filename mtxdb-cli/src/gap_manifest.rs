//! Durable record of a room's unresolved `prev_events`.
//!
//! Three facts stay separate so a policy decision can never masquerade as
//! evidence:
//!
//! - availability: what the queried servers said, or that nothing was asked;
//! - waiver: an operator chose not to chase a gap;
//! - anchor: a trusted state reference from outside the DAG.
//!
//! A waiver never makes state authoritative. An anchor always names whether it
//! supplies the state before or after its event, because `/state_ids` for an
//! event returns the state *before* it.
//!
//! Availability answers are scoped to one assessment and to the servers it
//! asked: `not_found` means "not found on these named servers in that
//! assessment", never "nobody has it". Older assessments are kept as history,
//! but the current status reads only the newest one, so a later all-404
//! assessment can supersede an earlier timeout. A gap-fill report that omits
//! an id because it found the event is not evidence of anything here. Only the
//! event arriving (in the import batch or already stored in the room) closes a
//! gap.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use mtxdb::interop::{
    event_record_id, gap_manifest_collection_id, gap_manifest_node_id, room_event_collection_id,
    EVENT_POOL, SERVER_INFO_POOL,
};
use mtxdb::{Database, NodeData, StorageEngine};
use rezzy::json::Value as OwnedValue;

/// Layout version. No compatibility is kept: an older record is refused.
const VERSION: u8 = 1;

/// Guards against a corrupt count turning a read into an allocation.
const MAX_GAPS: usize = 1_000_000;
const MAX_ANSWERS: usize = 1_024;

/// What one server said when asked for the missing event by id.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AnswerKind {
    /// The server returned the event, but mtxdb has not imported it yet.
    Found,
    /// A definite 404 from that server.
    NotFound,
    /// A 4xx other than 404. The server would not answer, so this says
    /// nothing about whether it holds the event.
    Refused,
    /// A timeout, 408, 429, 5xx, or network failure.
    Transient,
    /// The id was never requested by itself.
    Unanswered,
}

impl AnswerKind {
    /// Stable spelling shared with rezzy's gap-fill report.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Found => "found",
            Self::NotFound => "not_found",
            Self::Refused => "refused",
            Self::Transient => "transient",
            Self::Unanswered => "unanswered",
        }
    }

    const fn code(self) -> u8 {
        match self {
            Self::Found => 0,
            Self::NotFound => 1,
            Self::Refused => 2,
            Self::Transient => 3,
            Self::Unanswered => 4,
        }
    }

    const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Found),
            1 => Some(Self::NotFound),
            2 => Some(Self::Refused),
            3 => Some(Self::Transient),
            4 => Some(Self::Unanswered),
            _ => None,
        }
    }

    /// Whether this answer says the event is not held here.
    const fn is_definite_absence(self) -> bool {
        matches!(self, Self::NotFound)
    }
}

/// One server's answer about one gap, kept with the request outcome so a
/// refusal is not mistaken for absence.
///
/// Answers are evidence from a named server in a named assessment. They are
/// kept across assessments as history rather than overwritten, but the
/// *current* status reads only the latest assessment: what a server said last
/// year does not overrule what it said in the run being judged.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ServerAnswer {
    pub(crate) server: String,
    /// Label for the gap-fill run that produced this answer. Re-reading the
    /// same report replaces that assessment's answer for the server; a new
    /// assessment adds another record.
    pub(crate) run: String,
    /// Position of this assessment in ingest order, assigned by the ingest.
    /// The newest assessment is the highest `seq`, so a backwards clock step
    /// cannot make an older run current. `recorded_at_ms` breaks ties.
    pub(crate) seq: u64,
    /// When this answer was recorded, so the newest assessment and the
    /// status's `as_of` can be named without trusting vector order.
    pub(crate) recorded_at_ms: u64,
    pub(crate) kind: AnswerKind,
    pub(crate) status: Option<u16>,
    pub(crate) errcode: Option<String>,
}

/// What the queried servers said about a gap.
///
/// Absence is derived from the latest assessment, never stored as an
/// independent claim: it requires a definite 404 from every server that
/// assessment covered. A refusal, transient failure or unanswered query in
/// that assessment leaves the gap open. Earlier assessments stay here as
/// history and do not block a later all-404 assessment.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct Availability {
    pub(crate) answers: Vec<ServerAnswer>,
}

impl Availability {
    /// Every recorded answer in assessment order: by `seq`, then server. The
    /// stored order is not meaningful, because re-reading a run moves its
    /// answers to the end.
    pub(crate) fn history(&self) -> Vec<&ServerAnswer> {
        let mut answers: Vec<&ServerAnswer> = self.answers.iter().collect();
        answers.sort_by(|a, b| (a.seq, &a.server).cmp(&(b.seq, &b.server)));
        answers
    }

    /// The run label of the newest answer. Ties go to the answer recorded
    /// later, which is the one `record_answer` appended last.
    fn latest_run(&self) -> Option<&str> {
        let mut newest: Option<&ServerAnswer> = None;
        for answer in &self.answers {
            let key = (answer.seq, answer.recorded_at_ms);
            if newest.is_none_or(|current| key >= (current.seq, current.recorded_at_ms)) {
                newest = Some(answer);
            }
        }
        newest.map(|answer| answer.run.as_str())
    }

    /// The servers this assessment covered, taken from its answers.
    ///
    /// The ingest records one answer per gap for every entry in the report's
    /// `queried_servers`, using `Unanswered` when the report did not name the
    /// id. That keeps this set equal to `queried_servers`: a server the latest
    /// run named but never answered still counts and keeps the gap open, while
    /// a server dropped from that run stops counting.
    fn servers_in_run(&self, run: &str) -> BTreeSet<String> {
        self.answers
            .iter()
            .filter(|answer| answer.run == run)
            .map(|answer| answer.server.clone())
            .collect()
    }

    /// One server's answer in one assessment.
    fn answer_in_run(&self, server: &str, run: &str) -> Option<&ServerAnswer> {
        self.answers
            .iter()
            .find(|answer| answer.server == server && answer.run == run)
    }
}

/// An operator decision not to chase a gap. It never claims the gap is closed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Waiver {
    pub(crate) by: String,
    pub(crate) reason: String,
    pub(crate) recorded_at_ms: u64,
}

/// Whether an anchor supplies the state before or after its event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AnchorBoundary {
    Before,
    After,
}

impl AnchorBoundary {
    const fn code(self) -> u8 {
        match self {
            Self::Before => 1,
            Self::After => 2,
        }
    }

    const fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Before),
            2 => Some(Self::After),
            _ => None,
        }
    }
}

/// A trusted state reference from outside the DAG.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Anchor {
    pub(crate) event_id: String,
    pub(crate) boundary: AnchorBoundary,
    pub(crate) source: String,
    pub(crate) recorded_at_ms: u64,
}

/// Everything known about one unresolved prev.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct GapRecord {
    pub(crate) availability: Availability,
    pub(crate) waiver: Option<Waiver>,
    pub(crate) anchor: Option<Anchor>,
}

/// A gap's state, derived from its records. Never stored: a new answer, a
/// waiver or an anchor changes it without rewriting a status field.
///
/// Order matters. An anchor is the only fact that can complete state past a
/// gap, so it outranks everything. A waiver is an operator decision, so it
/// outranks mere evidence. `AbsentAmong` reads the latest assessment only; it
/// is still a gap and never closes one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GapStatus {
    Anchored,
    Waived,
    /// Every server in this one assessment answered a definite 404.
    ///
    /// The set names exactly the servers that assessment covered, and
    /// `as_of_ms` is the newest answer in it. It is not a claim that nobody
    /// holds the event, and it does not speak for servers the assessment did
    /// not ask. A later assessment replaces it; an earlier timeout in another
    /// assessment is history.
    AbsentAmong {
        servers: BTreeSet<String>,
        as_of_ms: u64,
    },
    Open,
}

impl GapRecord {
    fn open() -> Self {
        Self::default()
    }

    /// Derive the status from the latest assessment.
    ///
    /// See [`GapStatus`]. Use [`Self::status_for`] to judge a named
    /// assessment and server set instead.
    #[allow(
        dead_code,
        reason = "read by the strict-import gap check, which is not wired yet"
    )]
    pub(crate) fn status(&self) -> GapStatus {
        if self.anchor.is_some() {
            return GapStatus::Anchored;
        }
        if self.waiver.is_some() {
            return GapStatus::Waived;
        }
        let Some(run) = self.availability.latest_run() else {
            return GapStatus::Open;
        };
        let servers = self.availability.servers_in_run(run);
        self.status_for(run, &servers)
    }

    /// Derive the status from one assessment over an explicit server set.
    ///
    /// A server in the set with no answer in this assessment, or with any
    /// answer other than a definite 404, leaves the gap open. The set is
    /// explicit because "not found" is only meaningful for the sources that
    /// were actually asked.
    #[allow(
        dead_code,
        reason = "read by the strict-import gap check, which is not wired yet"
    )]
    pub(crate) fn status_for(&self, run: &str, servers: &BTreeSet<String>) -> GapStatus {
        if self.anchor.is_some() {
            return GapStatus::Anchored;
        }
        if self.waiver.is_some() {
            return GapStatus::Waived;
        }
        if servers.is_empty() {
            return GapStatus::Open;
        }
        let mut as_of_ms = 0;
        for server in servers {
            let Some(answer) = self.availability.answer_in_run(server, run) else {
                return GapStatus::Open;
            };
            if !answer.kind.is_definite_absence() {
                return GapStatus::Open;
            }
            as_of_ms = as_of_ms.max(answer.recorded_at_ms);
        }
        GapStatus::AbsentAmong {
            servers: servers.clone(),
            as_of_ms,
        }
    }
}

/// A room's gaps, keyed by the missing event id.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct GapManifest {
    pub(crate) gaps: BTreeMap<String, GapRecord>,
}

impl GapManifest {
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.gaps.is_empty()
    }

    /// Record a server answer. Evidence is kept per `(server, run)`.
    ///
    /// Re-reading the same assessment's report replaces that assessment's
    /// answer for the server, so duplicate ingestion does not pile up. A
    /// different assessment is appended: older answers stay as history, while
    /// the status reads only the newest assessment.
    pub(crate) fn record_answer(&mut self, gap: &str, answer: ServerAnswer) {
        let record = self.gaps.entry(gap.to_owned()).or_default();
        record
            .availability
            .answers
            .retain(|existing| !(existing.server == answer.server && existing.run == answer.run));
        record.availability.answers.push(answer);
    }
}

impl GapStatus {
    /// The short name used by `gaps list --status`.
    pub(crate) const fn name(&self) -> &'static str {
        match self {
            Self::Anchored => "anchored",
            Self::Waived => "waived",
            Self::AbsentAmong { .. } => "absent",
            Self::Open => "open",
        }
    }

    /// A one-line description. `absent` always names its servers: it is a
    /// claim about those sources only.
    pub(crate) fn describe(&self) -> String {
        match self {
            Self::AbsentAmong { servers, as_of_ms } => {
                let named: Vec<&str> = servers.iter().map(String::as_str).collect();
                format!("absent among {} (as of {as_of_ms} ms)", named.join(", "))
            }
            other => other.name().to_owned(),
        }
    }
}

impl GapManifest {
    /// Record an operator decision not to chase a tracked gap.
    ///
    /// A waiver is policy only. It never claims the gap is closed.
    ///
    /// # Errors
    /// Returns an error for a gap the manifest does not track (a waiver never
    /// creates one) or an empty `by` or `reason`.
    pub(crate) fn waive(
        &mut self,
        gap: &str,
        by: &str,
        reason: &str,
        recorded_at_ms: u64,
    ) -> Result<()> {
        if by.is_empty() || reason.is_empty() {
            bail!("a waiver needs both who made it and why");
        }
        let record = self
            .gaps
            .get_mut(gap)
            .with_context(|| format!("{gap} is not a tracked gap in this room"))?;
        record.waiver = Some(Waiver {
            by: by.to_owned(),
            reason: reason.to_owned(),
            recorded_at_ms,
        });
        Ok(())
    }

    /// Record a trusted state reference for a tracked gap. The boundary is
    /// required because `/state_ids` for an event is the state *before* it.
    ///
    /// # Errors
    /// Returns an error for an untracked gap or an empty `event_id` or `source`.
    pub(crate) fn anchor(
        &mut self,
        gap: &str,
        event_id: &str,
        boundary: AnchorBoundary,
        source: &str,
        recorded_at_ms: u64,
    ) -> Result<()> {
        if event_id.is_empty() || source.is_empty() {
            bail!("an anchor needs the event it describes and where the state came from");
        }
        let record = self
            .gaps
            .get_mut(gap)
            .with_context(|| format!("{gap} is not a tracked gap in this room"))?;
        record.anchor = Some(Anchor {
            event_id: event_id.to_owned(),
            boundary,
            source: source.to_owned(),
            recorded_at_ms,
        });
        Ok(())
    }
}

/// How strictly an import treats unresolved prevs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PrevGate {
    /// Record gaps and import anyway.
    Off,
    /// Fail unless every gap is anchored.
    Require,
    /// As `Require`, but an operator-waived gap also passes.
    RequireAllowWaived,
}

impl PrevGate {
    pub(crate) const fn from_flags(require: bool, allow_waived: bool) -> Self {
        match (require, allow_waived) {
            (false, _) => Self::Off,
            (true, false) => Self::Require,
            (true, true) => Self::RequireAllowWaived,
        }
    }

    /// Whether a gap with this derived status is acceptable.
    ///
    /// Only an anchor completes state past a gap. `AbsentAmong` is evidence
    /// about named servers, not completion, so it fails like `Open`. A waiver
    /// is accepted only when the operator opted in to waived gaps.
    const fn accepts(self, status: &GapStatus) -> bool {
        match status {
            GapStatus::Anchored => true,
            GapStatus::Waived => matches!(self, Self::RequireAllowWaived),
            GapStatus::AbsentAmong { .. } | GapStatus::Open => false,
        }
    }
}

/// The gaps in `manifest` that `gate` does not accept.
fn unaccepted(manifest: &GapManifest, gate: PrevGate) -> Vec<(&str, GapStatus)> {
    if gate == PrevGate::Off {
        return Vec::new();
    }
    manifest
        .gaps
        .iter()
        .map(|(id, record)| (id.as_str(), record.status()))
        .filter(|(_, status)| !gate.accepts(status))
        .collect()
}

/// One line describing tracked prev-gap history and whether the selected
/// strict-import gate accepts it. This does not make claims about other room
/// completeness dimensions such as signatures or auth chains. An anchor is
/// trusted operator-supplied input; this report does not independently verify it.
pub(crate) fn summarize(manifest: &GapManifest, gate: PrevGate) -> String {
    if manifest.gaps.is_empty() {
        return "prevs: complete (no tracked gaps)".to_owned();
    }

    let mut counts = [0_usize; 4];
    let mut absent = BTreeMap::<BTreeSet<String>, (usize, u64)>::new();
    for record in manifest.gaps.values() {
        let index = match record.status() {
            GapStatus::Open => 0,
            GapStatus::AbsentAmong { .. } => 1,
            GapStatus::Waived => 2,
            GapStatus::Anchored => 3,
        };
        counts[index] = counts[index].saturating_add(1);
    }

    // Keep the human report's order aligned with the gate's decision order.
    // Group absence by the assessed server set so repeated assessments do not
    // produce duplicate rows. The gaps in a group may have been assessed at
    // different times, so the single timestamp is the newest one written as an
    // upper bound: every assessment behind the row is no later than that.
    for record in manifest.gaps.values() {
        if let GapStatus::AbsentAmong { servers, as_of_ms } = record.status() {
            let (count, newest_as_of) = absent.entry(servers).or_default();
            *count = count.saturating_add(1);
            *newest_as_of = (*newest_as_of).max(as_of_ms);
        }
    }
    let mut parts = Vec::new();
    if counts[0] > 0 {
        parts.push(format!("{} open", counts[0]));
    }
    parts.extend(absent.into_iter().map(|(servers, (count, as_of_ms))| {
        let names: Vec<&str> = servers.iter().map(String::as_str).collect();
        format!(
            "{count} absent among {} (as of ≤ {as_of_ms} ms)",
            names.join(", ")
        )
    }));
    if counts[2] > 0 {
        parts.push(format!("{} waived", counts[2]));
    }
    if counts[3] > 0 {
        parts.push(format!("{} anchored", counts[3]));
    }

    let statuses: Vec<GapStatus> = manifest.gaps.values().map(GapRecord::status).collect();
    let gate_result = match gate {
        PrevGate::Off => "strict check not requested".to_owned(),
        _ if statuses.iter().all(|status| gate.accepts(status)) => "strict check passes".to_owned(),
        _ => "strict check fails".to_owned(),
    };

    let classification = if counts[0] == 0 && counts[1] == 0 && counts[2] == 0 {
        format!("gapped history, state known by anchor ({})", counts[3])
    } else if counts[0] == 0 && counts[1] == 0 && counts[3] == 0 {
        format!("gapped history, waived by operator ({})", counts[2])
    } else {
        format!("gapped ({})", parts.join(", "))
    };
    format!("prevs: {classification}; {gate_result}")
}

/// Print one line per touched room, so an import always says whether it left
/// gaps, with or without `--require-complete-prevs`.
///
/// # Errors
/// Returns a storage error from reading a manifest.
pub(crate) fn report_rooms(db: &Database, rooms: &BTreeSet<String>, gate: PrevGate) -> Result<()> {
    for room in rooms {
        eprintln!("  {room}: {}", summarize(&read(db, room)?, gate));
    }
    Ok(())
}

/// Fail an import that leaves a touched room with an unaccepted gap.
///
/// Every room in `rooms` is checked, and every failure is reported, not just
/// the first. Run it after the import has recorded its gaps and before the
/// import marker is committed.
///
/// # Errors
/// Returns an error naming each room and gap that failed, or a storage error
/// from reading a manifest.
pub(crate) fn check_complete_prevs(
    db: &Database,
    rooms: &BTreeSet<String>,
    gate: PrevGate,
) -> Result<()> {
    const SHOWN: usize = 5;
    if gate == PrevGate::Off {
        return Ok(());
    }
    let mut report = Vec::new();
    for room in rooms {
        let manifest = read(db, room)?;
        let failing = unaccepted(&manifest, gate);
        if failing.is_empty() {
            continue;
        }
        let mut lines = vec![format!("{room}: {} unresolved prev(s)", failing.len())];
        for (id, status) in failing.iter().take(SHOWN) {
            lines.push(format!("    {id}: {}", status.describe()));
        }
        if failing.len() > SHOWN {
            lines.push(format!(
                "    ... and {} more",
                failing.len().saturating_sub(SHOWN)
            ));
        }
        report.push(lines.join("\n"));
    }
    if report.is_empty() {
        return Ok(());
    }
    bail!(
        "--require-complete-prevs: the import leaves gaps that are not anchored{}:\n{}",
        if gate == PrevGate::RequireAllowWaived {
            " or waived"
        } else {
            " (use --allow-waived-gaps to accept waived ones)"
        },
        report.join("\n")
    )
}

/// What an ingest did, so the caller can say what it did and did not do.
#[derive(Debug, Default, Eq, PartialEq)]
pub(crate) struct IngestSummary {
    /// Tracked gaps that received this run's answers.
    pub(crate) recorded: usize,
    /// Ids the report covers that the manifest does not track.
    pub(crate) untracked: Vec<String>,
    /// Tracked gaps the report says were found. They stay open until the event
    /// is imported; a report cannot close a gap.
    pub(crate) found_not_yet_imported: Vec<String>,
}

impl GapManifest {
    /// The next assessment position: one past the highest recorded.
    fn next_seq(&self) -> u64 {
        self.gaps
            .values()
            .flat_map(|record| &record.availability.answers)
            .map(|answer| answer.seq)
            .max()
            .map_or(1, |seq| seq.saturating_add(1))
    }

    /// The position already given to `run`, so re-reading its report cannot
    /// promote it above a newer assessment.
    fn seq_of_run(&self, run: &str) -> Option<u64> {
        self.gaps
            .values()
            .flat_map(|record| &record.availability.answers)
            .filter(|answer| answer.run == run)
            .map(|answer| answer.seq)
            .max()
    }
}

/// The servers a report says it asked. A report that asked no one assessed
/// nothing.
fn parse_queried(report: &OwnedValue) -> Result<BTreeSet<String>> {
    let queried: BTreeSet<String> = report
        .get("queried_servers")
        .and_then(OwnedValue::as_array)
        .and_then(|servers| {
            servers
                .iter()
                .map(|item| item.as_str().filter(|s| !s.is_empty()).map(str::to_owned))
                .collect()
        })
        .context("report has no valid `queried_servers`")?;
    if queried.is_empty() {
        bail!("report names no queried servers, so nothing was assessed");
    }
    Ok(queried)
}

/// Every `(id, server)` answer a report carries: the still-missing outcomes and
/// the `resolved` ids. Each must name a queried server and appear once.
fn parse_outcomes(
    report: &OwnedValue,
    queried: &BTreeSet<String>,
) -> Result<BTreeMap<String, BTreeMap<String, ServerAnswerParts>>> {
    let mut parsed: BTreeMap<String, BTreeMap<String, ServerAnswerParts>> = BTreeMap::new();
    let mut insert = |id: &str, server: &str, parts: ServerAnswerParts| -> Result<()> {
        if !queried.contains(server) {
            bail!("outcome for {id} names {server}, which the run did not query");
        }
        let previous = parsed
            .entry(id.to_owned())
            .or_default()
            .insert(server.to_owned(), parts);
        if previous.is_some() {
            bail!("duplicate outcome for {id} on {server}");
        }
        Ok(())
    };
    let outcomes = report
        .get("missing_prev_outcomes")
        .and_then(OwnedValue::as_array)
        .context("report has no `missing_prev_outcomes`")?;
    for entry in outcomes {
        let field = |name: &str| entry.get(name).and_then(OwnedValue::as_str);
        let id = field("id").context("outcome without an `id`")?;
        let server = field("server").context("outcome without a `server`")?;
        let kind = match field("outcome") {
            Some("not_found") => AnswerKind::NotFound,
            Some("refused") => AnswerKind::Refused,
            Some("transient") => AnswerKind::Transient,
            Some("unanswered") => AnswerKind::Unanswered,
            other => bail!("outcome for {id} on {server} is not recognized: {other:?}"),
        };
        let status = entry
            .get("status")
            .and_then(OwnedValue::as_u64)
            .and_then(|n| u16::try_from(n).ok());
        insert(
            id,
            server,
            (kind, status, field("errcode").map(str::to_owned)),
        )?;
    }
    let resolved = report
        .get("resolved")
        .and_then(OwnedValue::as_array)
        .context("report has no `resolved` list")?;
    for entry in resolved {
        let field = |name: &str| entry.get(name).and_then(OwnedValue::as_str);
        let id = field("id").context("resolved entry without an `id`")?;
        let server = field("server").context("resolved entry without a `server`")?;
        if field("outcome") != Some("found") {
            bail!("resolved id {id} on {server} does not have outcome `found`");
        }
        insert(id, server, (AnswerKind::Found, None, None))?;
    }
    Ok(parsed)
}

/// Record a `rezzy federation gap-fill` report as one assessment named `run`.
///
/// Every tracked gap the report covers gets one answer for each server in
/// `queried_servers`, `Unanswered` where the report names no outcome, so the
/// assessed set equals what the run actually asked. Only definite answers from
/// a named server are evidence; omission is not, and `resolved` closes
/// nothing. The whole report is validated before the manifest is touched.
///
/// # Errors
/// Returns an error for a report without `queried_servers` or
/// `missing_prev_outcomes`, an outcome from a server the run did not name, a
/// duplicate or unrecognized outcome, or an empty `run`.
pub(crate) fn ingest_report(
    manifest: &mut GapManifest,
    report: &OwnedValue,
    run: &str,
    now_ms: u64,
) -> Result<IngestSummary> {
    if run.is_empty() {
        bail!("a gap-fill run needs a non-empty label");
    }
    let queried = parse_queried(report)?;
    let parsed = parse_outcomes(report, &queried)?;

    let seq = manifest
        .seq_of_run(run)
        .unwrap_or_else(|| manifest.next_seq());
    let mut summary = IngestSummary::default();
    let reported_ids: BTreeSet<String> = parsed.keys().cloned().collect();
    for id in &reported_ids {
        if !manifest.gaps.contains_key(id) {
            summary.untracked.push(id.clone());
        }
    }
    let tracked_ids: Vec<String> = manifest.gaps.keys().cloned().collect();
    for id in tracked_ids {
        let by_server = parsed.get(&id);
        for server in &queried {
            let (kind, status, errcode) = by_server
                .and_then(|by_server| by_server.get(server))
                .cloned()
                .unwrap_or((AnswerKind::Unanswered, None, None));
            manifest.record_answer(
                &id,
                ServerAnswer {
                    server: server.clone(),
                    run: run.to_owned(),
                    seq,
                    recorded_at_ms: now_ms,
                    kind,
                    status,
                    errcode,
                },
            );
        }
        summary.recorded = summary.recorded.saturating_add(1);
    }
    summary.found_not_yet_imported = parsed
        .iter()
        .filter(|(id, by_server)| {
            manifest.gaps.contains_key(*id)
                && by_server
                    .values()
                    .any(|(kind, _, _)| *kind == AnswerKind::Found)
        })
        .map(|(id, _)| id.clone())
        .collect();
    Ok(summary)
}

type ServerAnswerParts = (AnswerKind, Option<u16>, Option<String>);

/// Read a room's manifest, or an empty one if none is stored.
///
/// # Errors
/// Propagates storage errors and refuses a malformed record.
pub(crate) fn read(db: &Database, room_id: &str) -> Result<GapManifest> {
    let record = db
        .pool(SERVER_INFO_POOL)
        .get(
            &gap_manifest_collection_id(room_id.as_bytes()),
            &gap_manifest_node_id(),
        )
        .context("reading the gap manifest")?;
    match record {
        Some(data) => decode(&data.bytes),
        None => Ok(GapManifest::default()),
    }
}

/// Write a room's manifest, removing the record when no gaps remain.
///
/// # Errors
/// Propagates encoding, staging and commit errors.
pub(crate) fn write(db: &Database, room_id: &str, manifest: &GapManifest) -> Result<()> {
    let collection = gap_manifest_collection_id(room_id.as_bytes());
    let node_id = gap_manifest_node_id();
    let tx = db.begin_transaction();
    if manifest.is_empty() {
        tx.remove(SERVER_INFO_POOL, collection, node_id)
            .context("removing the gap manifest")?;
    } else {
        tx.put(
            SERVER_INFO_POOL,
            collection,
            node_id,
            &NodeData::new(bytes::Bytes::from(encode(manifest)?)),
        )
        .context("writing the gap manifest")?;
    }
    tx.commit().context("committing the gap manifest")?;
    Ok(())
}

/// Record the gaps visible in `events` and clear gaps that are now resolved.
///
/// Existing records keep their answers, waiver and anchor; only newly seen
/// gaps start open. A gap is resolved only when its event is in this batch or
/// already stored in the room: a report that says the event was found, or
/// omits it from a missing list, does not close the gap. Presence in the store
/// is what lets the state walk resolve the prev.
///
/// # Errors
/// Propagates storage errors from the read, resolution check and write.
pub(crate) fn record_import_gaps(
    db: &Database,
    room_id: &str,
    events: &[OwnedValue],
    auth_chain: &[OwnedValue],
) -> Result<()> {
    let mut manifest = read(db, room_id)?;
    let known: HashSet<String> = events
        .iter()
        .chain(auth_chain)
        .filter_map(event_id)
        .map(str::to_owned)
        .collect();
    let mut missing = BTreeSet::new();
    for event in events {
        for parent in prev_ids(event) {
            if !known.contains(parent) {
                missing.insert(parent.to_owned());
            }
        }
    }
    let mut resolved = BTreeSet::new();
    for id in manifest.gaps.keys() {
        if known.contains(id) || room_holds(db, room_id, id)? {
            resolved.insert(id.clone());
        }
    }
    manifest.gaps.retain(|id, _| !resolved.contains(id));
    for id in missing {
        // A prev absent from this batch may already be stored in the room
        // (an export re-import); that is not a gap.
        if room_holds(db, room_id, &id)? {
            continue;
        }
        manifest.gaps.entry(id).or_insert_with(GapRecord::open);
    }
    write(db, room_id, &manifest)
}

/// Timestamp for a waiver or anchor recorded now.
#[must_use]
pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
        })
}

/// Whether the room's event collection already holds `event_id`.
fn room_holds(db: &Database, room_id: &str, event_id: &str) -> Result<bool> {
    // A failed lookup must not read as "not held": that would record a false
    // gap, or keep a resolved one.
    Ok(db
        .pool(EVENT_POOL)
        .get(
            &room_event_collection_id(room_id.as_bytes()),
            &event_record_id(event_id.as_bytes()),
        )?
        .is_some())
}

fn event_id(event: &OwnedValue) -> Option<&str> {
    event
        .get("event_id")
        .and_then(OwnedValue::as_str)
        .filter(|id| !id.is_empty())
}

fn prev_ids(event: &OwnedValue) -> Vec<&str> {
    event
        .get("prev_events")
        .and_then(OwnedValue::as_array)
        .map(|references| {
            references
                .iter()
                .filter_map(|reference| match reference {
                    OwnedValue::String(id) => Some(id.as_str()),
                    OwnedValue::Array(parts) => parts.first().and_then(OwnedValue::as_str),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Encode a manifest as `GVM1`.
///
/// # Errors
/// Returns an error for a field or count that does not fit its prefix.
pub(crate) fn encode(manifest: &GapManifest) -> Result<Vec<u8>> {
    let too_large = || anyhow::anyhow!("gap manifest field is too large");
    let mut out = vec![VERSION];
    out.extend_from_slice(
        &u32::try_from(manifest.gaps.len())
            .map_err(|_| too_large())?
            .to_le_bytes(),
    );
    for (id, record) in &manifest.gaps {
        put_str(&mut out, id)?;
        out.extend_from_slice(
            &u32::try_from(record.availability.answers.len())
                .map_err(|_| too_large())?
                .to_le_bytes(),
        );
        for answer in &record.availability.answers {
            put_str(&mut out, &answer.server)?;
            put_str(&mut out, &answer.run)?;
            out.extend_from_slice(&answer.seq.to_le_bytes());
            out.extend_from_slice(&answer.recorded_at_ms.to_le_bytes());
            out.push(answer.kind.code());
            match answer.status {
                Some(status) => {
                    out.push(1);
                    out.extend_from_slice(&status.to_le_bytes());
                }
                None => out.push(0),
            }
            put_opt_str(&mut out, answer.errcode.as_deref())?;
        }
        match &record.waiver {
            Some(waiver) => {
                out.push(1);
                put_str(&mut out, &waiver.by)?;
                put_str(&mut out, &waiver.reason)?;
                out.extend_from_slice(&waiver.recorded_at_ms.to_le_bytes());
            }
            None => out.push(0),
        }
        match &record.anchor {
            Some(anchor) => {
                out.push(anchor.boundary.code());
                put_str(&mut out, &anchor.event_id)?;
                put_str(&mut out, &anchor.source)?;
                out.extend_from_slice(&anchor.recorded_at_ms.to_le_bytes());
            }
            None => out.push(0),
        }
    }
    Ok(out)
}

/// Decode a `GVM1` record. Trailing bytes are corrupt, not ignored.
///
/// # Errors
/// Returns an error for a truncated record, an unknown version or answer code,
/// a count over the guard, or trailing bytes.
pub(crate) fn decode(bytes: &[u8]) -> Result<GapManifest> {
    let mut reader = Reader::new(bytes);
    let version = reader.u8()?;
    if version != VERSION {
        bail!("unknown gap manifest version {version}");
    }
    let count = reader.len_prefixed(MAX_GAPS, "gap count")?;
    let mut gaps = BTreeMap::new();
    for _ in 0..count {
        let id = reader.string()?;
        let answers = reader.len_prefixed(MAX_ANSWERS, "answer count")?;
        let mut record = GapRecord::default();
        for _ in 0..answers {
            let server = reader.string()?;
            let run = reader.string()?;
            let seq = reader.u64()?;
            let recorded_at_ms = reader.u64()?;
            let kind = AnswerKind::from_code(reader.u8()?)
                .ok_or_else(|| anyhow::anyhow!("unknown gap answer code"))?;
            let status = match reader.u8()? {
                0 => None,
                1 => Some(reader.u16()?),
                _ => bail!("invalid gap answer status flag"),
            };
            let errcode = reader.opt_string()?;
            record.availability.answers.push(ServerAnswer {
                server,
                run,
                seq,
                recorded_at_ms,
                kind,
                status,
                errcode,
            });
        }
        if reader.u8()? == 1 {
            record.waiver = Some(Waiver {
                by: reader.string()?,
                reason: reader.string()?,
                recorded_at_ms: reader.u64()?,
            });
        }
        let boundary = reader.u8()?;
        if boundary != 0 {
            record.anchor = Some(Anchor {
                boundary: AnchorBoundary::from_code(boundary)
                    .ok_or_else(|| anyhow::anyhow!("unknown anchor boundary"))?,
                event_id: reader.string()?,
                source: reader.string()?,
                recorded_at_ms: reader.u64()?,
            });
        }
        gaps.insert(id, record);
    }
    if !reader.is_empty() {
        bail!("gap manifest has trailing bytes");
    }
    Ok(GapManifest { gaps })
}

fn put_str(out: &mut Vec<u8>, value: &str) -> Result<()> {
    out.extend_from_slice(
        &u32::try_from(value.len())
            .map_err(|_| anyhow::anyhow!("gap manifest field is too large"))?
            .to_le_bytes(),
    );
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn put_opt_str(out: &mut Vec<u8>, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        out.push(1);
        put_str(out, value)
    } else {
        out.push(0);
        Ok(())
    }
}

/// Bounds-checked cursor over a manifest record.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(len)
            .ok_or_else(|| anyhow::anyhow!("gap manifest length overflow"))?;
        let slice = self
            .bytes
            .get(self.at..end)
            .ok_or_else(|| anyhow::anyhow!("gap manifest is truncated"))?;
        self.at = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn len_prefixed(&mut self, max: usize, what: &str) -> Result<usize> {
        let count = usize::try_from(self.u32()?).unwrap_or(usize::MAX);
        if count > max {
            bail!("gap manifest {what} {count} exceeds {max}");
        }
        Ok(count)
    }

    fn string(&mut self) -> Result<String> {
        let len = self.len_prefixed(self.bytes.len(), "string length")?;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).context("gap manifest string is not UTF-8")
    }

    fn opt_string(&mut self) -> Result<Option<String>> {
        match self.u8()? {
            0 => Ok(None),
            1 => self.string().map(Some),
            _ => bail!("invalid gap manifest optional flag"),
        }
    }

    fn is_empty(&self) -> bool {
        self.at == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answer_at(server: &str, run: &str, at_ms: u64, kind: AnswerKind) -> ServerAnswer {
        ServerAnswer {
            server: server.to_owned(),
            run: run.to_owned(),
            seq: 0,
            recorded_at_ms: at_ms,
            kind,
            status: match kind {
                AnswerKind::Unanswered => None,
                _ => Some(404),
            },
            errcode: None,
        }
    }

    fn answer(server: &str, run: &str, kind: AnswerKind) -> ServerAnswer {
        answer_at(server, run, 1, kind)
    }

    fn absent(servers: &[&str], as_of_ms: u64) -> GapStatus {
        GapStatus::AbsentAmong {
            servers: servers.iter().map(|server| (*server).to_owned()).collect(),
            as_of_ms,
        }
    }

    fn sample() -> GapManifest {
        let mut manifest = GapManifest::default();
        manifest.record_answer(
            "$absent",
            answer("matrix.example", "run-1", AnswerKind::NotFound),
        );
        manifest.record_answer(
            "$waived",
            answer("a.example", "run-1", AnswerKind::NotFound),
        );
        manifest.gaps.get_mut("$waived").unwrap().waiver = Some(Waiver {
            by: "operator".to_owned(),
            reason: "no source has it".to_owned(),
            recorded_at_ms: 7,
        });
        let mut open = GapRecord::open();
        open.availability.answers = vec![
            answer("a.example", "run-1", AnswerKind::NotFound),
            answer("b.example", "run-1", AnswerKind::Transient),
        ];
        open.anchor = Some(Anchor {
            event_id: "$later".to_owned(),
            boundary: AnchorBoundary::Before,
            source: "snapshot".to_owned(),
            recorded_at_ms: 9,
        });
        manifest.gaps.insert("$open".to_owned(), open);
        manifest
    }

    #[test]
    fn a_manifest_round_trips_every_separate_fact() {
        let manifest = sample();
        let decoded = decode(&encode(&manifest).unwrap()).unwrap();
        assert_eq!(decoded, manifest);
        assert_eq!(
            decoded.gaps["$absent"].status(),
            absent(&["matrix.example"], 1)
        );
        assert_eq!(decoded.gaps["$waived"].status(), GapStatus::Waived);
        assert_eq!(decoded.gaps["$open"].status(), GapStatus::Anchored);
    }

    #[test]
    fn absence_reads_only_the_latest_assessment() {
        let mut gap = GapRecord::open();
        assert_eq!(
            gap.status(),
            GapStatus::Open,
            "nothing asked is not absence"
        );

        gap.availability.answers = vec![answer("a", "run-1", AnswerKind::NotFound)];
        assert_eq!(gap.status(), absent(&["a"], 1));

        gap.availability
            .answers
            .push(answer("b", "run-1", AnswerKind::Refused));
        assert_eq!(gap.status(), GapStatus::Open, "a refusal is not absence");

        gap.availability.answers = vec![
            answer("a", "run-1", AnswerKind::NotFound),
            answer("b", "run-1", AnswerKind::Transient),
        ];
        assert_eq!(gap.status(), GapStatus::Open, "a 5xx is not absence");

        gap.availability.answers = vec![answer("a", "run-1", AnswerKind::Unanswered)];
        assert_eq!(
            gap.status(),
            GapStatus::Open,
            "an unsent query is not absence"
        );
    }

    #[test]
    fn a_gap_status_is_derived_from_its_facts_in_priority_order() {
        let mut gap = GapRecord::open();
        assert_eq!(gap.status(), GapStatus::Open, "no facts is open");

        gap.availability.answers = vec![answer("a", "run-1", AnswerKind::NotFound)];
        assert_eq!(
            gap.status(),
            absent(&["a"], 1),
            "an answer and only definite 404s is absent among those servers"
        );

        gap.availability
            .answers
            .push(answer("b", "run-1", AnswerKind::Transient));
        assert_eq!(
            gap.status(),
            GapStatus::Open,
            "one non-404 leaves the gap open"
        );

        gap.waiver = Some(Waiver {
            by: "operator".to_owned(),
            reason: "accepted".to_owned(),
            recorded_at_ms: 1,
        });
        assert_eq!(
            gap.status(),
            GapStatus::Waived,
            "a waiver outranks availability"
        );

        gap.anchor = Some(Anchor {
            event_id: "$later".to_owned(),
            boundary: AnchorBoundary::Before,
            source: "snapshot".to_owned(),
            recorded_at_ms: 2,
        });
        assert_eq!(
            gap.status(),
            GapStatus::Anchored,
            "an anchor outranks a waiver"
        );
    }

    #[test]
    fn a_later_all_not_found_assessment_restores_absence_but_history_survives() {
        let mut manifest = GapManifest::default();
        manifest.record_answer("$gap", answer_at("a", "run-1", 1, AnswerKind::NotFound));
        manifest.record_answer("$gap", answer_at("b", "run-1", 1, AnswerKind::Transient));
        assert_eq!(manifest.gaps["$gap"].status(), GapStatus::Open);

        // A later assessment asks both servers again and both answer 404.
        manifest.record_answer("$gap", answer_at("a", "run-2", 2, AnswerKind::NotFound));
        manifest.record_answer("$gap", answer_at("b", "run-2", 2, AnswerKind::NotFound));
        assert_eq!(manifest.gaps["$gap"].status(), absent(&["a", "b"], 2));
        assert_eq!(
            manifest.gaps["$gap"].availability.answers.len(),
            4,
            "earlier assessments are kept as history"
        );

        // A newer assessment with one timeout is open again.
        manifest.record_answer("$gap", answer_at("a", "run-3", 3, AnswerKind::Transient));
        manifest.record_answer("$gap", answer_at("b", "run-3", 3, AnswerKind::NotFound));
        assert_eq!(manifest.gaps["$gap"].status(), GapStatus::Open);
    }

    #[test]
    fn status_is_scoped_to_the_servers_the_assessment_asked() {
        let mut gap = GapRecord::open();
        gap.availability.answers = vec![
            answer_at("a", "run-1", 1, AnswerKind::NotFound),
            answer_at("b", "run-1", 1, AnswerKind::Transient),
        ];
        assert_eq!(gap.status(), GapStatus::Open);
        assert_eq!(
            gap.status_for("run-1", &BTreeSet::from(["a".to_owned()])),
            absent(&["a"], 1),
            "absence is only claimed for the servers actually asked"
        );
        assert_eq!(
            gap.status_for("run-2", &BTreeSet::from(["a".to_owned()])),
            GapStatus::Open,
            "an answer from another assessment does not count for this one"
        );
        assert_eq!(
            gap.status_for("run-1", &BTreeSet::from(["a".to_owned(), "b".to_owned()])),
            GapStatus::Open
        );
        assert_eq!(
            gap.status_for("run-1", &BTreeSet::from(["c".to_owned()])),
            GapStatus::Open,
            "a server with no answer in this assessment is not absent"
        );
        assert_eq!(
            gap.status_for("run-1", &BTreeSet::new()),
            GapStatus::Open,
            "an empty assessment decides nothing"
        );
    }

    #[test]
    fn the_newest_timestamp_picks_the_assessment_not_vector_order() {
        let mut manifest = GapManifest::default();
        manifest.record_answer("$gap", answer_at("a", "run-2", 2, AnswerKind::NotFound));
        // An older assessment ingested later must not become current.
        manifest.record_answer("$gap", answer_at("a", "run-1", 1, AnswerKind::Transient));
        assert_eq!(
            manifest.gaps["$gap"].status(),
            absent(&["a"], 2),
            "the timestamp, not insertion order, names the latest assessment"
        );
    }

    #[test]
    fn re_reading_one_assessment_replaces_that_servers_answer() {
        let mut manifest = GapManifest::default();
        manifest.record_answer("$gap", answer_at("a", "run-1", 1, AnswerKind::NotFound));
        manifest.record_answer("$gap", answer_at("a", "run-1", 1, AnswerKind::Transient));
        assert_eq!(manifest.gaps["$gap"].availability.answers.len(), 1);
        assert_eq!(manifest.gaps["$gap"].status(), GapStatus::Open);

        manifest.record_answer("$gap", answer_at("a", "run-2", 2, AnswerKind::NotFound));
        assert_eq!(manifest.gaps["$gap"].status(), absent(&["a"], 2));
        assert_eq!(manifest.gaps["$gap"].availability.answers.len(), 2);
    }

    #[test]
    fn a_server_dropped_from_the_latest_run_stops_counting() {
        let mut manifest = GapManifest::default();
        manifest.record_answer("$gap", answer_at("a", "run-1", 1, AnswerKind::NotFound));
        manifest.record_answer("$gap", answer_at("b", "run-1", 1, AnswerKind::NotFound));
        assert_eq!(manifest.gaps["$gap"].status(), absent(&["a", "b"], 1));

        // The newest run asks only `a`; `b`'s stale 404 no longer carries.
        manifest.record_answer("$gap", answer_at("a", "run-2", 2, AnswerKind::NotFound));
        assert_eq!(manifest.gaps["$gap"].status(), absent(&["a"], 2));
    }

    #[test]
    fn a_server_named_but_unanswered_in_the_latest_run_keeps_the_gap_open() {
        let mut gap = GapRecord::open();
        gap.availability.answers = vec![
            answer_at("a", "run-1", 1, AnswerKind::NotFound),
            answer_at("b", "run-1", 1, AnswerKind::Unanswered),
        ];
        assert_eq!(
            gap.status(),
            GapStatus::Open,
            "a queried server with no answer is not a 404"
        );
    }

    #[test]
    fn truncated_or_trailing_bytes_are_corrupt() {
        let bytes = encode(&sample()).unwrap();
        assert!(decode(&bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(decode(&trailing).is_err());
        trailing[0] = 2;
        assert!(decode(&trailing).is_err());
    }

    #[test]
    fn an_empty_manifest_writes_no_record() {
        let encoded = encode(&GapManifest::default()).unwrap();
        assert!(decode(&encoded).unwrap().is_empty());
    }

    fn report(json: &str) -> OwnedValue {
        OwnedValue::parse_bytes(json.as_bytes()).expect("test report is JSON")
    }

    fn tracked(ids: &[&str]) -> GapManifest {
        let mut manifest = GapManifest::default();
        for id in ids {
            manifest.gaps.insert((*id).to_owned(), GapRecord::open());
        }
        manifest
    }

    #[test]
    fn ingest_answers_every_queried_server_and_marks_the_unnamed_unanswered() {
        let mut manifest = tracked(&["$a", "$b"]);
        let summary = ingest_report(
            &mut manifest,
            &report(
                r#"{"queried_servers":["x.test","y.test"],"resolved":[],
                    "missing_prev_outcomes":[
                      {"id":"$a","server":"x.test","outcome":"not_found","status":404},
                      {"id":"$a","server":"y.test","outcome":"not_found","status":404},
                      {"id":"$b","server":"x.test","outcome":"not_found","status":404}]}"#,
            ),
            "run-1",
            10,
        )
        .unwrap();
        assert_eq!(summary.recorded, 2);
        assert_eq!(
            manifest.gaps["$a"].status(),
            absent(&["x.test", "y.test"], 10)
        );
        // y.test was queried but named nothing for $b: that is not a 404.
        assert_eq!(manifest.gaps["$b"].availability.answers.len(), 2);
        assert_eq!(manifest.gaps["$b"].status(), GapStatus::Open);
    }

    #[test]
    fn a_bad_report_changes_nothing() {
        let mut manifest = tracked(&["$a"]);
        let before = manifest.clone();
        for bad in [
            // an outcome from a server the run never named
            r#"{"queried_servers":["x.test"],"missing_prev_outcomes":[
                {"id":"$a","server":"z.test","outcome":"not_found"}]}"#,
            // `found` cannot be an answer for a still-missing prev
            r#"{"queried_servers":["x.test"],"missing_prev_outcomes":[
                {"id":"$a","server":"x.test","outcome":"found"}]}"#,
            // duplicate (id, server)
            r#"{"queried_servers":["x.test"],"missing_prev_outcomes":[
                {"id":"$a","server":"x.test","outcome":"not_found"},
                {"id":"$a","server":"x.test","outcome":"transient"}]}"#,
            r#"{"queried_servers":[],"missing_prev_outcomes":[]}"#,
            r#"{"missing_prev_outcomes":[]}"#,
            r#"{"queried_servers":["x.test"]}"#,
        ] {
            assert!(
                ingest_report(&mut manifest, &report(bad), "r", 1).is_err(),
                "{bad}"
            );
            assert_eq!(manifest, before, "{bad}");
        }
        assert!(ingest_report(
            &mut manifest,
            &report(r#"{"queried_servers":["x"],"missing_prev_outcomes":[]}"#),
            "",
            1
        )
        .is_err());
    }

    #[test]
    fn a_report_never_creates_or_closes_a_gap() {
        let mut manifest = tracked(&["$a", "$found"]);
        let summary = ingest_report(
            &mut manifest,
            &report(
                r#"{"queried_servers":["x.test"],
                    "resolved":[{"id":"$found","server":"x.test","outcome":"found"}],
                    "missing_prev_outcomes":[
                      {"id":"$other","server":"x.test","outcome":"not_found"}]}"#,
            ),
            "run-1",
            1,
        )
        .unwrap();
        assert_eq!(summary.untracked, vec!["$other".to_owned()]);
        assert_eq!(summary.found_not_yet_imported, vec!["$found".to_owned()]);
        assert!(
            !manifest.gaps.contains_key("$other"),
            "ingest creates no gap"
        );
        assert!(
            manifest.gaps.contains_key("$found"),
            "a found id stays a gap until its event is imported"
        );
    }

    #[test]
    fn ingest_order_beats_the_clock_and_a_re_read_cannot_promote_an_old_run() {
        let mut manifest = tracked(&["$a"]);
        let run = |outcome: &str| {
            report(&format!(
                r#"{{"queried_servers":["x.test"],"resolved":[],"missing_prev_outcomes":[
                    {{"id":"$a","server":"x.test","outcome":"{outcome}"}}]}}"#
            ))
        };
        ingest_report(&mut manifest, &run("transient"), "old", 900).unwrap();
        // The clock stepped backwards: the newer assessment has a smaller time.
        ingest_report(&mut manifest, &run("not_found"), "new", 100).unwrap();
        assert_eq!(manifest.gaps["$a"].status(), absent(&["x.test"], 100));
        // Re-reading the old report keeps its place in the order.
        ingest_report(&mut manifest, &run("transient"), "old", 2000).unwrap();
        assert_eq!(manifest.gaps["$a"].status(), absent(&["x.test"], 100));
        let bytes = encode(&manifest).unwrap();
        assert_eq!(decode(&bytes).unwrap(), manifest, "seq survives the codec");
    }

    #[test]
    fn a_found_id_is_recorded_but_keeps_the_gap_open_and_resolved_is_required() {
        let mut manifest = tracked(&["$a"]);
        ingest_report(
            &mut manifest,
            &report(
                r#"{"queried_servers":["x.test"],"missing_prev_outcomes":[],
                    "resolved":[{"id":"$a","server":"x.test","outcome":"found"}]}"#,
            ),
            "run-1",
            5,
        )
        .unwrap();
        let record = &manifest.gaps["$a"];
        assert_eq!(record.availability.answers[0].kind, AnswerKind::Found);
        assert_eq!(
            record.status(),
            GapStatus::Open,
            "found is not absence, and the event is not imported yet"
        );
        let before = manifest.clone();
        for bad in [
            r#"{"queried_servers":["x.test"],"missing_prev_outcomes":[]}"#,
            r#"{"queried_servers":["x.test"],"missing_prev_outcomes":[],
                "resolved":[{"id":"$a","server":"z.test","outcome":"found"}]}"#,
            r#"{"queried_servers":["x.test"],"missing_prev_outcomes":[],
                "resolved":[{"id":"$a","server":"x.test","outcome":"not_found"}]}"#,
        ] {
            assert!(ingest_report(&mut manifest, &report(bad), "run-2", 6).is_err());
            assert_eq!(manifest, before);
        }
    }

    #[test]
    fn waive_and_anchor_need_a_tracked_gap_and_never_create_one() {
        let mut manifest = tracked(&["$a"]);
        assert!(manifest.waive("$nope", "me", "why", 1).is_err());
        assert!(manifest
            .anchor("$nope", "$e", AnchorBoundary::Before, "src", 1)
            .is_err());
        assert!(!manifest.gaps.contains_key("$nope"));
        assert!(manifest.waive("$a", "", "why", 1).is_err());
        assert!(manifest.waive("$a", "me", "", 1).is_err());
        assert!(manifest
            .anchor("$a", "", AnchorBoundary::After, "src", 1)
            .is_err());
        assert_eq!(manifest.gaps["$a"].status(), GapStatus::Open);

        manifest.waive("$a", "me", "accepted", 2).unwrap();
        assert_eq!(manifest.gaps["$a"].status(), GapStatus::Waived);
        manifest
            .anchor("$a", "$e", AnchorBoundary::Before, "state_ids", 3)
            .unwrap();
        assert_eq!(manifest.gaps["$a"].status(), GapStatus::Anchored);
        assert_eq!(
            manifest.gaps["$a"].anchor.as_ref().unwrap().boundary,
            AnchorBoundary::Before
        );
    }

    #[test]
    fn a_status_names_the_servers_an_absence_covers() {
        assert_eq!(GapStatus::Open.describe(), "open");
        let text = absent(&["a.test", "b.test"], 9).describe();
        assert!(text.contains("a.test, b.test") && text.starts_with("absent among"));
        assert_eq!(absent(&["a.test"], 9).name(), "absent");
    }

    #[test]
    fn history_follows_assessment_order_and_names_match_the_report() {
        let mut manifest = tracked(&["$a"]);
        let run = |outcome: &str, server: &str| {
            report(&format!(
                r#"{{"queried_servers":["{server}"],"resolved":[],"missing_prev_outcomes":[
                    {{"id":"$a","server":"{server}","outcome":"{outcome}"}}]}}"#
            ))
        };
        ingest_report(&mut manifest, &run("transient", "x.test"), "old", 1).unwrap();
        ingest_report(&mut manifest, &run("not_found", "x.test"), "new", 2).unwrap();
        // Re-reading the old run moves its answer to the end of the stored vector.
        ingest_report(&mut manifest, &run("transient", "x.test"), "old", 3).unwrap();
        let runs: Vec<&str> = manifest.gaps["$a"]
            .availability
            .history()
            .iter()
            .map(|answer| answer.run.as_str())
            .collect();
        assert_eq!(runs, ["old", "new"]);
        for kind in [
            AnswerKind::Found,
            AnswerKind::NotFound,
            AnswerKind::Refused,
            AnswerKind::Transient,
            AnswerKind::Unanswered,
        ] {
            assert!(!kind.name().chars().any(char::is_uppercase), "{kind:?}");
        }
        assert_eq!(AnswerKind::NotFound.name(), "not_found");
    }

    #[test]
    fn the_gate_accepts_only_anchored_and_optionally_waived_gaps() {
        let mut manifest = tracked(&["$open", "$absent", "$waived", "$anchored"]);
        ingest_report(
            &mut manifest,
            &report(
                r#"{"queried_servers":["x.test"],"resolved":[],"missing_prev_outcomes":[
                    {"id":"$absent","server":"x.test","outcome":"not_found"}]}"#,
            ),
            "run",
            1,
        )
        .unwrap();
        manifest.waive("$waived", "me", "ok", 1).unwrap();
        manifest
            .anchor("$anchored", "$e", AnchorBoundary::Before, "src", 1)
            .unwrap();
        let ids = |gate| -> Vec<&str> {
            unaccepted(&manifest, gate)
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        };
        assert_eq!(ids(PrevGate::Off), Vec::<&str>::new());
        assert_eq!(ids(PrevGate::Require), ["$absent", "$open", "$waived"]);
        assert_eq!(ids(PrevGate::RequireAllowWaived), ["$absent", "$open"]);
        assert_eq!(
            unaccepted(&GapManifest::default(), PrevGate::Require),
            Vec::new()
        );
        assert_eq!(PrevGate::from_flags(false, true), PrevGate::Off);
    }

    #[test]
    fn a_room_summary_reports_gap_history_and_the_gate_result() {
        assert_eq!(
            summarize(&GapManifest::default(), PrevGate::Require),
            "prevs: complete (no tracked gaps)"
        );
        let mut manifest = tracked(&["$a", "$b", "$c"]);
        manifest.waive("$b", "me", "ok", 1).unwrap();
        manifest
            .anchor("$c", "$e", AnchorBoundary::After, "src", 1)
            .unwrap();
        assert_eq!(
            summarize(&manifest, PrevGate::Require),
            "prevs: gapped (1 open, 1 waived, 1 anchored); strict check fails"
        );
        let mut anchored = tracked(&["$a"]);
        anchored
            .anchor("$a", "$e", AnchorBoundary::Before, "src", 1)
            .unwrap();
        assert_eq!(
            summarize(&anchored, PrevGate::Require),
            "prevs: gapped history, state known by anchor (1); strict check passes"
        );
        let mut waived = tracked(&["$a"]);
        waived.waive("$a", "me", "ok", 1).unwrap();
        assert_eq!(
            summarize(&waived, PrevGate::RequireAllowWaived),
            "prevs: gapped history, waived by operator (1); strict check passes"
        );
    }

    #[test]
    fn room_summary_groups_absence_by_server_set_at_newest_time_as_upper_bound() {
        let mut manifest = tracked(&["$a", "$b"]);
        for (id, run, seq, at_ms) in [("$a", "run-a", 1, 10), ("$b", "run-b", 2, 20)] {
            let mut answer = answer_at("peer.test", run, at_ms, AnswerKind::NotFound);
            answer.seq = seq;
            manifest.record_answer(id, answer);
        }
        assert_eq!(
            summarize(&manifest, PrevGate::Require),
            "prevs: gapped (2 absent among peer.test (as of ≤ 20 ms)); strict check fails"
        );
    }
}
