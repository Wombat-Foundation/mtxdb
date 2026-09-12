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

use mtxdb_core::storage::{NodeData, StorageEngine};
use mtxdb_core::PackfileStorage;

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
    pub fn open(path: &str, pool: Option<String>) -> Result<MdbStorage, JsValue> {
        let layout = mtxdb_core::DatabaseLayout::open(PathBuf::from(path))
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let shard_type = match pool.as_deref() {
            None | Some("event-dag") => mtxdb_core::ShardType::EventDag,
            Some("state") => mtxdb_core::ShardType::State,
            Some("auth-chain") => mtxdb_core::ShardType::AuthChain,
            Some(pool) => return Err(JsValue::from_str(&format!("unsupported pool: {pool}"))),
        };
        let pool_dir = layout.pool_dir(shard_type)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let storage = PackfileStorage::open(pool_dir)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner: storage })
    }

    /// Store a node. Both `collection_id` and `node_id` must be 16-byte arrays.
    pub fn put(&self, collection_id: &[u8], node_id: &[u8], data: &[u8]) -> Result<(), JsValue> {
        let collection = as_id(collection_id, "collection_id")?;
        let id = as_id(node_id, "node_id")?;
        let node_data = NodeData::new(bytes::Bytes::copy_from_slice(data));
        self.inner
            .put(&collection, &id, &node_data)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Fetch a node. Returns `null` if not found.
    pub fn get(&self, collection_id: &[u8], node_id: &[u8]) -> Result<Option<Vec<u8>>, JsValue> {
        let collection = as_id(collection_id, "collection_id")?;
        let id = as_id(node_id, "node_id")?;
        match self
            .inner
            .get(&collection, &id)
            .map_err(|e| JsValue::from_str(&e.to_string()))?
        {
            Some(data) => Ok(Some(data.bytes.to_vec())),
            None => Ok(None),
        }
    }

    /// Delete all data for a collection.
    pub fn delete_collection(&self, collection_id: &[u8]) -> Result<(), JsValue> {
        let collection = as_id(collection_id, "collection_id")?;
        self.inner
            .delete_collection(&collection)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Sync all packfiles to disk.
    pub fn sync(&self) -> Result<(), JsValue> {
        self.inner
            .sync()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

fn as_id(bytes: &[u8], label: &str) -> Result<[u8; 16], JsValue> {
    if bytes.len() != 16 {
        return Err(JsValue::from_str(&format!("{label} must be 16 bytes")));
    }
    let mut id = [0u8; 16];
    id.copy_from_slice(bytes);
    Ok(id)
}
