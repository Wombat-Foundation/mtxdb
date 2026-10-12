//! Per-event disposition, state views, and the resolver/trust policy that
//! state-group identity commits to.
//!
//! Two separate things live here:
//!
//! - [`EventDisposition`] is the *claim an import carries* about an event.
//!   mtxdb does not compute it by verifying signatures; verification is an
//!   ingestion stage that runs earlier (see `ValidationProfile` in the
//!   library). State resolution only ever reads the resulting disposition.
//! - [`resolution_policy`] turns a room version plus the importer's trust mode
//!   into the [`StateGroupDerivation`] hashed into every instance id, so a
//!   group resolved by one algorithm can never collide with a group produced
//!   by another, and an unverified import can never masquerade as a verified
//!   one.

use mtxdb::state_group::{StateGroupDerivation, StateGroupResolver, StateGroupTrust};
use rezzy::json::Value as OwnedValue;
use rezzy::StateResVersion;

/// What an event's own validation claimed about it.
///
/// This is the input state resolution consumes; the resolver never re-derives
/// it. The three values map directly onto the spec's three outcomes:
///
/// - **`Accepted`** passes authorization against its own auth chain and
///   against current state.
/// - **`Rejected`** fails authorization against its own auth chain. It stays
///   in the event/auth DAG (descendants may still reference it) but
///   contributes no state and never enters resolution as a state update.
/// - **`SoftFailed`** passes against its own historical state but fails against
///   current state. It stays in the DAG *and* in federated state, and
///   participates in state resolution like any other event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum EventDisposition {
    /// Passed whatever validation the importer applied.
    Accepted,
    /// Failed authorization; retained in the DAG, contributes no state.
    Rejected,
    /// Passed historically, failed against current state; stays in federated
    /// state.
    SoftFailed,
}

/// Whether an event carries a top-level boolean `true` flag named `flag`
/// (`"rejected"` or `"soft_failed"`). Servers annotate this differently, so
/// leading underscores are ignored and `-` equals `_`: `__soft-failed`,
/// `_soft-failed`, `soft-failed`, and `soft_failed` all name the same flag.
pub(crate) fn has_event_flag(event: &OwnedValue, flag: &str) -> bool {
    let OwnedValue::Object(fields) = event else {
        return false;
    };
    fields.iter().any(|(key, value)| {
        matches!(value, OwnedValue::Bool(true))
            && key.trim_start_matches('_').replace('-', "_") == flag
    })
}

/// Read an event's disposition from its import annotations.
///
/// A rejected flag wins over a soft-failed one: an event that fails
/// authorization against its own auth chain is rejected regardless of any
/// soft-fail annotation it also carries.
#[must_use]
pub(crate) fn event_disposition(event: &OwnedValue) -> EventDisposition {
    if has_event_flag(event, "rejected") {
        EventDisposition::Rejected
    } else if has_event_flag(event, "soft_failed") {
        EventDisposition::SoftFailed
    } else {
        EventDisposition::Accepted
    }
}

/// Which view of room state a state group describes.
///
/// Federated state is what servers agree on: a soft-failed event is still part
/// of it. A client never sees soft-failed events, so its state can differ, and
/// therefore so can its state group. A rejected event fails authorization and
/// belongs to neither view. Redactions change an event's content but not its
/// ID, and a state group is keyed by `(type, state_key, event_id)`, so they
/// affect neither view's group.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum StateView {
    /// What servers agree on; includes soft-failed events.
    Federated,
    /// What a client is shown; drops soft-failed events.
    #[allow(
        dead_code,
        reason = "the client-visible view, for callers that need it"
    )]
    Client,
}

impl StateView {
    /// Whether this view drops the event from the state set it tracks.
    #[must_use]
    pub(crate) fn excludes(self, event: &OwnedValue) -> bool {
        match event_disposition(event) {
            EventDisposition::Rejected => true,
            EventDisposition::SoftFailed => self == Self::Client,
            EventDisposition::Accepted => false,
        }
    }
}

/// How a state group was derived: which resolver ran, and what the importer
/// had proven about its inputs beforehand.
///
/// `rezzy_version` is the algorithm passed to rezzy's resolver. It is `None`
/// exactly when `derivation.resolver` is
/// [`StateGroupResolver::FirstParent`], i.e. when no real resolution ran
/// because the room version is unrecognized — there are no auth rules to
/// apply without one, so the merge keeps the first-parent rule rather
/// than guessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResolutionPolicy {
    /// The identity-bearing half of the derivation.
    pub(crate) derivation: StateGroupDerivation,
    /// The rezzy algorithm to run, or `None` for the first-parent merge.
    pub(crate) rezzy_version: Option<StateResVersion>,
}

impl ResolutionPolicy {
    /// Whether a real state-resolution algorithm is available.
    #[must_use]
    pub(crate) const fn resolvable(self) -> bool {
        self.rezzy_version.is_some()
    }
}

/// Tag a rezzy algorithm version. Every [`StateResVersion`] has a matching
/// [`StateGroupResolver`] so no algorithm can be folded into another.
#[must_use]
const fn resolver_for(version: StateResVersion) -> StateGroupResolver {
    match version {
        StateResVersion::V1 => StateGroupResolver::V1,
        StateResVersion::V2 => StateGroupResolver::V2,
        StateResVersion::V2_1 => StateGroupResolver::V2_1,
        StateResVersion::V2_1_1 => StateGroupResolver::V2_1_1,
        StateResVersion::V2_2 => StateGroupResolver::V2_2,
        StateResVersion::V3 => StateGroupResolver::V3,
    }
}

/// Build the full derivation for an import: the room version decides the
/// resolver, the importer states the trust claim.
///
/// An unrecognized room version degrades to
/// [`StateGroupResolver::FirstParent`] with no rezzy version, which is exactly
/// "keep the first-parent merge": mtxdb cannot run auth rules it does not know.
#[must_use]
pub(crate) fn resolution_policy(room_version: &str, trust: StateGroupTrust) -> ResolutionPolicy {
    let Some(version) = StateResVersion::from_room_version(room_version) else {
        return ResolutionPolicy {
            derivation: StateGroupDerivation::new(StateGroupResolver::FirstParent, trust),
            rezzy_version: None,
        };
    };
    ResolutionPolicy {
        derivation: StateGroupDerivation::new(resolver_for(version), trust),
        rezzy_version: Some(version),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(json: &str) -> OwnedValue {
        let bytes = json.as_bytes().to_vec();
        OwnedValue::parse_bytes(&bytes).expect("valid JSON fixture")
    }

    #[test]
    fn disposition_reads_every_annotation_spelling() {
        for key in ["__rejected", "_rejected", "rejected"] {
            let event = value(&format!(r#"{{"{key}":true,"type":"m.room.name"}}"#));
            assert_eq!(
                event_disposition(&event),
                EventDisposition::Rejected,
                "{key}"
            );
        }
        for key in [
            "__soft-failed",
            "_soft-failed",
            "soft-failed",
            "soft_failed",
        ] {
            let event = value(&format!(r#"{{"{key}":true,"type":"m.room.name"}}"#));
            assert_eq!(
                event_disposition(&event),
                EventDisposition::SoftFailed,
                "{key}"
            );
        }
        let plain = value(r#"{"type":"m.room.name"}"#);
        assert_eq!(event_disposition(&plain), EventDisposition::Accepted);
        // `false` is not a flag, and a rejected flag outranks a soft-failed one.
        let not_flagged = value(r#"{"__rejected":false,"type":"m.room.name"}"#);
        assert_eq!(event_disposition(&not_flagged), EventDisposition::Accepted);
        let both = value(r#"{"rejected":true,"soft_failed":true,"type":"m.room.name"}"#);
        assert_eq!(event_disposition(&both), EventDisposition::Rejected);
    }

    #[test]
    fn views_filter_on_disposition() {
        let federated = StateView::Federated;
        let client = StateView::Client;
        let accepted = value(r#"{"type":"m.room.name"}"#);
        let rejected = value(r#"{"rejected":true,"type":"m.room.name"}"#);
        let soft_failed = value(r#"{"soft_failed":true,"type":"m.room.name"}"#);

        for view in [federated, client] {
            assert!(!view.excludes(&accepted));
            assert!(view.excludes(&rejected));
        }
        assert!(!federated.excludes(&soft_failed));
        assert!(client.excludes(&soft_failed));
    }

    #[test]
    fn room_versions_map_to_their_spec_resolver() {
        assert_eq!(
            resolution_policy("1", StateGroupTrust::TrustedImport).rezzy_version,
            Some(StateResVersion::V1)
        );
        for version in ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11"] {
            let policy = resolution_policy(version, StateGroupTrust::TrustedImport);
            assert_eq!(
                policy.rezzy_version,
                Some(StateResVersion::V2),
                "room version {version}"
            );
            assert_eq!(
                policy.derivation.resolver,
                StateGroupResolver::V2,
                "room version {version}"
            );
        }
        {
            let policy = resolution_policy("12", StateGroupTrust::TrustedImport);
            assert_eq!(policy.rezzy_version, Some(StateResVersion::V2_1));
            assert_eq!(policy.derivation.resolver, StateGroupResolver::V2_1);
        }
        // 12.1 is a separate algorithm, not stock v2.1.
        let policy = resolution_policy("12.1", StateGroupTrust::TrustedImport);
        assert_eq!(policy.rezzy_version, Some(StateResVersion::V2_1_1));
        assert_eq!(policy.derivation.resolver, StateGroupResolver::V2_1_1);
    }

    #[test]
    fn unknown_room_versions_decline_to_resolve() {
        let policy = resolution_policy("0.9", StateGroupTrust::TrustedImport);
        assert!(!policy.resolvable());
        assert_eq!(policy.derivation.resolver, StateGroupResolver::FirstParent);
    }

    #[test]
    fn trust_mode_is_carried_into_the_derivation() {
        let trusted = resolution_policy("10", StateGroupTrust::TrustedImport);
        let validated = resolution_policy("10", StateGroupTrust::ValidatedImport);
        assert_eq!(trusted.derivation.resolver, validated.derivation.resolver);
        assert_ne!(trusted.derivation, validated.derivation);
    }
}
