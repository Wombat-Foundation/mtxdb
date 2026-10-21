use super::*;

/// The numbers are stored in every verdict and written in every export. A
/// change here is a breaking format change, so the full table is pinned.
#[test]
fn rejection_code_numbers_are_pinned() {
    let table: [(RejectionCode, u8, &str); 12] = [
        (
            RejectionCode::SignatureVerificationFailed,
            0,
            "signature_verification_failed",
        ),
        (RejectionCode::InvalidPduFormat, 1, "invalid_pdu_format"),
        (
            RejectionCode::DependsOnRejectedAuthEvent,
            2,
            "depends_on_rejected_auth_event",
        ),
        (RejectionCode::PrevEventsRejected, 3, "prev_events_rejected"),
        (RejectionCode::MissingAuthEvent, 4, "missing_auth_event"),
        (
            RejectionCode::DuplicateAuthEventKey,
            5,
            "duplicate_auth_event_key",
        ),
        (RejectionCode::MissingCreateEvent, 6, "missing_create_event"),
        (RejectionCode::AuthCheckFailed, 7, "auth_check_failed"),
        (
            RejectionCode::StructurallyInvalidInGetMissingEvents,
            8,
            "structurally_invalid_in_get_missing_events",
        ),
        (
            RejectionCode::PrevEventUnknownStateIdsFailed,
            9,
            "prev_event_unknown_state_ids_failed",
        ),
        (
            RejectionCode::StateResolutionFailedWithPrevsPresent,
            10,
            "state_resolution_failed_with_prevs_present",
        ),
        (RejectionCode::Unknown, 11, "unknown"),
    ];
    assert_eq!(table.len(), RejectionCode::ALL.len());
    for ((code, number, tag), listed) in table.into_iter().zip(RejectionCode::ALL) {
        assert_eq!(code, listed, "ALL is in number order");
        assert_eq!(code.to_u8(), number);
        assert_eq!(RejectionCode::from_u8(number), code);
        assert_eq!(code.tag(), tag);
    }
}

#[test]
fn soft_fail_code_numbers_are_pinned() {
    let table: [(SoftFailCode, u8, &str); 4] = [
        (SoftFailCode::AuthCheckFailed, 0, "auth_check_failed"),
        (SoftFailCode::Imported, 1, "imported"),
        (SoftFailCode::Manual, 2, "manual"),
        (SoftFailCode::Unknown, 3, "unknown"),
    ];
    assert_eq!(table.len(), SoftFailCode::ALL.len());
    for ((code, number, tag), listed) in table.into_iter().zip(SoftFailCode::ALL) {
        assert_eq!(code, listed, "ALL is in number order");
        assert_eq!(code.to_u8(), number);
        assert_eq!(SoftFailCode::from_u8(number), code);
        assert_eq!(code.tag(), tag);
    }
}

#[test]
fn unknown_numbers_read_back_as_unknown() {
    for number in 12..=u8::MAX {
        assert_eq!(RejectionCode::from_u8(number), RejectionCode::Unknown);
    }
    for number in 4..=u8::MAX {
        assert_eq!(SoftFailCode::from_u8(number), SoftFailCode::Unknown);
    }
}

#[test]
fn reason_strings_parse_with_or_without_detail() {
    for code in RejectionCode::ALL {
        if code == RejectionCode::Unknown {
            assert_eq!(RejectionCode::parse(code.tag()), None);
            continue;
        }
        assert_eq!(RejectionCode::parse(code.tag()), Some(code));
        assert_eq!(
            RejectionCode::parse(&format!("{}: because", code.tag())),
            Some(code)
        );
    }
    for code in SoftFailCode::ALL {
        if code == SoftFailCode::Unknown {
            assert_eq!(SoftFailCode::parse(code.tag()), None);
            continue;
        }
        assert_eq!(SoftFailCode::parse(code.tag()), Some(code));
        assert_eq!(
            SoftFailCode::parse(&format!("{}: x", code.tag())),
            Some(code)
        );
    }
    assert_eq!(RejectionCode::parse("not a reason"), None);
    assert_eq!(SoftFailCode::parse(""), None);
}

/// The spelling of the wire format is the contract between a homeserver's
/// exporter and `mtxdb import`; pin every name.
#[test]
fn wire_names_are_pinned() {
    assert_eq!(EXPORT_FORMAT, "mtxdb-export/v1");
    assert_eq!(
        [tag::HEADER, tag::EVENT, tag::END, tag::SERVER_KEYS],
        ["header", "event", "end", "server_keys"]
    );
    assert_eq!(
        [
            field::T,
            field::FORMAT,
            field::SERVER,
            field::ROOM_ID,
            field::EVENT_ID,
            field::PREV_EVENTS,
            field::AUTH_EVENTS,
            field::OUTLIER,
            field::SOFT_FAILED,
            field::SOFT_FAIL_CODE,
            field::REJECTED,
            field::REJECTION_CODE,
            field::REDACTED_BY,
            field::PDU,
            field::EVENTS,
            field::ORIGIN,
            field::KIND,
            field::PAYLOAD,
        ],
        [
            "t",
            "format",
            "server",
            "room_id",
            "event_id",
            "prev_events",
            "auth_events",
            "outlier",
            "soft_failed",
            "soft_fail_code",
            "rejected",
            "rejection_code",
            "redacted_by",
            "pdu",
            "events",
            "origin",
            "kind",
            "payload",
        ]
    );
}
