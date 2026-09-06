//! C FFI bindings for mtxdb.
//!
//! Provides a C-callable API for using mtxdb from C, C++, Python, Go,
//! Swift, and other languages via FFI. All complex types are opaque
//! pointers; the caller must free with the provided destroy functions.

use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;
use std::slice;

use mtxdb::storage::{NodeData, StorageEngine};
use mtxdb::PackfileStorage;

/// Opaque handle to a PackfileStorage instance.
pub struct MdbStorage {
    inner: PackfileStorage,
}

/// Opaque handle to node data.
pub struct MdbNodeData {
    inner: NodeData,
}

/// Read a 16-byte ID from a raw pointer.
///
/// # Safety
/// `ptr` must point to at least 16 readable bytes.
unsafe fn read_id(ptr: *const u8) -> [u8; 16] {
    let slice = unsafe { slice::from_raw_parts(ptr, 16) };
    let mut id = [0u8; 16];
    id.copy_from_slice(slice);
    id
}

/// Error codes returned by FFI functions.
#[repr(C)]
pub enum MdbError {
    Ok = 0,
    Io = 1,
    NotFound = 2,
    InvalidInput = 3,
    Internal = 4,
}

/// Create a new storage instance at the given path.
///
/// # Safety
/// `path` must be a valid null-terminated UTF-8 C string.
///
/// # Returns
/// Null on error, otherwise an opaque handle. Caller must destroy with
/// `mdb_storage_destroy`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdb_storage_open(path: *const c_char) -> *mut MdbStorage {
    let c_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    let storage = match PackfileStorage::open(std::path::PathBuf::from(c_str)) {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    Box::into_raw(Box::new(MdbStorage { inner: storage }))
}

/// Destroy a storage instance and release all resources.
///
/// # Safety
/// `handle` must have been returned by `mdb_storage_open` and not
/// previously destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdb_storage_destroy(handle: *mut MdbStorage) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// Store a node. Returns `MdbError::Ok` on success.
///
/// # Safety
/// - `handle` must be a valid pointer from `mdb_storage_open`.
/// - `room_id` must point to at least 16 bytes.
/// - `node_id` must point to at least 16 bytes.
/// - `data` / `data_len` must reference a valid byte buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdb_put(
    handle: *mut MdbStorage,
    room_id: *const u8,
    node_id: *const u8,
    data: *const u8,
    data_len: usize,
) -> MdbError {
    let Some(storage) = (unsafe { handle.as_ref() }) else {
        return MdbError::InvalidInput;
    };
    let room = unsafe { read_id(room_id) };
    let id = unsafe { read_id(node_id) };
    let bytes = unsafe { slice::from_raw_parts(data, data_len) };
    let node_data = NodeData::new(bytes::Bytes::copy_from_slice(bytes));
    match storage.inner.put(&room, &id, &node_data) {
        Ok(()) => MdbError::Ok,
        Err(_) => MdbError::Io,
    }
}

/// Fetch a node. Returns null if not found.
///
/// The caller must destroy the returned handle with `mdb_node_data_destroy`.
///
/// # Safety
/// - `handle` must be a valid pointer from `mdb_storage_open`.
/// - `room_id` must point to at least 16 bytes.
/// - `node_id` must point to at least 16 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdb_get(
    handle: *mut MdbStorage,
    room_id: *const u8,
    node_id: *const u8,
) -> *mut MdbNodeData {
    let Some(storage) = (unsafe { handle.as_ref() }) else {
        return ptr::null_mut();
    };
    let room = unsafe { read_id(room_id) };
    let id = unsafe { read_id(node_id) };
    match storage.inner.get(&room, &id) {
        Ok(Some(data)) => Box::into_raw(Box::new(MdbNodeData { inner: data })),
        _ => ptr::null_mut(),
    }
}

/// Get the raw bytes of a node data handle.
///
/// Returns a pointer to the internal buffer and writes the length to
/// `*out_len`. The pointer is valid as long as the node data handle
/// exists.
///
/// # Safety
/// - `node` must be a valid pointer from `mdb_get`.
/// - `out_len` must be a valid mutable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdb_node_bytes(
    node: *const MdbNodeData,
    out_len: *mut usize,
) -> *const u8 {
    let Some(data) = (unsafe { node.as_ref() }) else {
        unsafe { *out_len = 0 };
        return ptr::null();
    };
    unsafe { *out_len = data.inner.bytes.len() };
    data.inner.bytes.as_ptr()
}

/// Destroy a node data handle.
///
/// # Safety
/// `node` must have been returned by `mdb_get` and not previously destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdb_node_data_destroy(node: *mut MdbNodeData) {
    if !node.is_null() {
        unsafe { drop(Box::from_raw(node)) };
    }
}

/// Sync all packfiles to disk.
///
/// # Safety
/// `handle` must be a valid pointer from `mdb_storage_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdb_sync(handle: *mut MdbStorage) -> MdbError {
    let Some(storage) = (unsafe { handle.as_ref() }) else {
        return MdbError::InvalidInput;
    };
    match storage.inner.sync() {
        Ok(()) => MdbError::Ok,
        Err(_) => MdbError::Io,
    }
}

/// Delete all data for a room.
///
/// # Safety
/// - `handle` must be a valid pointer from `mdb_storage_open`.
/// - `room_id` must point to at least 16 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdb_delete_room(handle: *mut MdbStorage, room_id: *const u8) -> MdbError {
    let Some(storage) = (unsafe { handle.as_ref() }) else {
        return MdbError::InvalidInput;
    };
    let room = unsafe { read_id(room_id) };
    match storage.inner.delete_room(&room) {
        Ok(()) => MdbError::Ok,
        Err(_) => MdbError::Io,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_ffi_roundtrip() {
        let dir = std::env::temp_dir().join("mdb_ffi_test");
        std::fs::create_dir_all(&dir).unwrap();

        let path = CString::new(dir.to_str().unwrap()).unwrap();
        let handle = unsafe { mdb_storage_open(path.as_ptr()) };
        assert!(!handle.is_null());

        let room = [0x01u8; 16];
        let id = [0x42u8; 16];
        let data = b"hello from ffi";

        let result = unsafe {
            mdb_put(
                handle,
                room.as_ptr(),
                id.as_ptr(),
                data.as_ptr(),
                data.len(),
            )
        };
        assert!(matches!(result, MdbError::Ok));

        let node = unsafe { mdb_get(handle, room.as_ptr(), id.as_ptr()) };
        assert!(!node.is_null());

        let mut out_len = 0usize;
        let bytes = unsafe { mdb_node_bytes(node, &mut out_len) };
        assert_eq!(out_len, data.len());
        let slice = unsafe { std::slice::from_raw_parts(bytes, out_len) };
        assert_eq!(slice, data);

        unsafe {
            mdb_node_data_destroy(node);
            mdb_storage_destroy(handle);
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
