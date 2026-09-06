//! WebAssembly bindings for mtxdb.
//!
//! **Target:** `wasm32-wasip1` or `wasm32-wasip2` — these targets provide
//! a WASI-compatible filesystem that mtxdb's mmap-based storage engine
//! can use directly.
//!
//! **Browser wasm** (`wasm32-unknown-unknown`) is not yet supported:
//! mtxdb relies on `memmap2` and `std::fs`, which require OS-level file
//! I/O. A browser backend using IndexedDB or OPFS would be a separate
//! effort.

use std::path::PathBuf;

use wasm_bindgen::prelude::*;

use mtxdb::storage::{NodeData, StorageEngine};
use mtxdb::PackfileStorage;

/// JavaScript-accessible handle to a mtxdb storage instance.
#[wasm_bindgen]
pub struct MdbStorage {
    inner: PackfileStorage,
}

#[wasm_bindgen]
impl MdbStorage {
    /// Open or create a storage instance at the given filesystem path.
    ///
    /// Only works on WASI targets with a mounted filesystem.
    #[wasm_bindgen(constructor)]
    pub fn open(path: &str) -> Result<MdbStorage, JsValue> {
        let storage = PackfileStorage::open(PathBuf::from(path))
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner: storage })
    }

    /// Store a node. Both `room_id` and `node_id` must be 16-byte arrays.
    pub fn put(&self, room_id: &[u8], node_id: &[u8], data: &[u8]) -> Result<(), JsValue> {
        let room = as_room_id(room_id)?;
        let id = as_node_id(node_id)?;
        let node_data = NodeData::new(bytes::Bytes::copy_from_slice(data));
        self.inner
            .put(&room, &id, &node_data)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Fetch a node. Returns `null` if not found.
    pub fn get(&self, room_id: &[u8], node_id: &[u8]) -> Result<Option<Vec<u8>>, JsValue> {
        let room = as_room_id(room_id)?;
        let id = as_node_id(node_id)?;
        match self
            .inner
            .get(&room, &id)
            .map_err(|e| JsValue::from_str(&e.to_string()))?
        {
            Some(data) => Ok(Some(data.bytes.to_vec())),
            None => Ok(None),
        }
    }

    /// Delete all data for a room.
    pub fn delete_room(&self, room_id: &[u8]) -> Result<(), JsValue> {
        let room = as_room_id(room_id)?;
        self.inner
            .delete_room(&room)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Sync all packfiles to disk.
    pub fn sync(&self) -> Result<(), JsValue> {
        self.inner
            .sync()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

fn as_room_id(bytes: &[u8]) -> Result<[u8; 16], JsValue> {
    if bytes.len() != 16 {
        return Err(JsValue::from_str("room_id must be 16 bytes"));
    }
    let mut id = [0u8; 16];
    id.copy_from_slice(bytes);
    Ok(id)
}

fn as_node_id(bytes: &[u8]) -> Result<[u8; 16], JsValue> {
    if bytes.len() != 16 {
        return Err(JsValue::from_str("node_id must be 16 bytes"));
    }
    let mut id = [0u8; 16];
    id.copy_from_slice(bytes);
    Ok(id)
}
