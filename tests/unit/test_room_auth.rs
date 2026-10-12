#![allow(clippy::tests_outside_test_module)]

//! The façade's error surface: one conversion from the storage taxonomy, with
//! the stale-generation case carrying both generations.

use std::error::Error;

use mtxdb::room_auth::RoomAuthError;
use mtxdb::storage::StorageError;

#[test]
fn a_stale_generation_keeps_both_generations() {
    let converted = RoomAuthError::from(StorageError::StaleGeneration {
        generation: 1,
        current: Some(3),
    });
    assert!(
        matches!(
            converted,
            RoomAuthError::StaleGeneration {
                pinned: 1,
                current: Some(3)
            }
        ),
        "{converted:?}"
    );
    assert_eq!(
        converted.to_string(),
        "pinned closure generation 1 is stale; current generation is 3"
    );
}

#[test]
fn a_purged_room_has_no_current_generation() {
    let converted = RoomAuthError::from(StorageError::StaleGeneration {
        generation: 2,
        current: None,
    });
    assert!(
        matches!(
            converted,
            RoomAuthError::StaleGeneration {
                pinned: 2,
                current: None
            }
        ),
        "{converted:?}"
    );
    assert!(
        converted.to_string().contains("no generation is published"),
        "{converted}"
    );
}

#[test]
fn corrupt_storage_data_becomes_corruption() {
    let converted = RoomAuthError::from(StorageError::Corrupt("bad head".to_owned()));
    assert!(
        matches!(&converted, RoomAuthError::Corruption(message) if message == "bad head"),
        "{converted:?}"
    );
}

#[test]
fn other_storage_errors_are_wrapped_with_their_source() {
    let converted = RoomAuthError::from(StorageError::Io(std::io::Error::other("disk")));
    assert!(
        matches!(converted, RoomAuthError::Storage(_)),
        "{converted:?}"
    );
    assert!(
        converted.source().is_some(),
        "a wrapped storage error exposes its cause"
    );
}
