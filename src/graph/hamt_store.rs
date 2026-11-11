//! Persisted, content-addressed HAMTs in an mtxdb collection.
//!
//! Generic over any key and value types with a
//! [`HamtCodec`](crate::hamt::HamtCodec) encoding. Each
//! internal node is an immutable `MTHN` record stored under the first 16 bytes
//! of its 32-byte structural hash ([`crate::state_hamt_node_id`]); reads verify
//! every loaded node against the full hash, so a truncated-id collision is an
//! error, never a silently wrong tree.
//!
//! Build trees with [`crate::hamt::build_hamt`]: `rezzy`'s HAMT routes keys by
//! their encoded bytes by default, so node bytes depend only on the logical
//! content and the structural key, and every read here routes the same way.
//!
//! Root records, collection choice, and what a key or value means are left to
//! the caller; [`crate::state_hamt`] is the Matrix room-state layer built on
//! this module.

use std::collections::HashSet;
use std::hash::Hash;
use std::sync::Arc;

use rezzy::hamt::{
    HamtCodec, HamtNode, HamtReadError, NodeRef, PersistedHamtReader, PersistedInternalNode,
    StructuralHash,
};

use crate::storage::{NodeData, StorageEngine, StorageError};
use crate::{DatabaseTransaction, PackfileStorage, ShardType};

/// Loads one encoded node from a collection by structural hash.
pub type NodeLoader<'a> =
    Box<dyn FnMut(&StructuralHash) -> Result<Option<Vec<u8>>, StorageError> + 'a>;

/// A verifying reader over one collection's nodes. It caches decoded nodes for
/// its lifetime, so hold one across related reads.
pub type HamtReader<'a, K, V> = PersistedHamtReader<'a, K, V, StorageError, NodeLoader<'a>>;

/// Why a persisted HAMT read failed.
pub type HamtStoreError = HamtReadError<StorageError>;

/// Encode every node of a fully resolved tree exactly once, children before
/// parents, as `(structural_hash, MTHN record)`.
///
/// Lazy children are skipped: they are already persisted.
#[must_use]
pub fn encode_nodes<K, V>(root: &Arc<HamtNode<K, V>>) -> Vec<(StructuralHash, Vec<u8>)>
where
    K: HamtCodec + Clone,
    V: HamtCodec + Clone,
{
    fn walk<K, V>(
        node: &Arc<HamtNode<K, V>>,
        seen: &mut HashSet<StructuralHash>,
        out: &mut Vec<(StructuralHash, Vec<u8>)>,
    ) where
        K: HamtCodec + Clone,
        V: HamtCodec + Clone,
    {
        if !seen.insert(node.structural_hash) {
            return;
        }
        for child in &node.children {
            if let NodeRef::Resolved(child_node) = child {
                walk(child_node, seen, out);
            }
        }
        let persisted: PersistedInternalNode<K, V> = node.as_ref().into();
        out.push((node.structural_hash, persisted.encode_v1()));
    }

    let mut out = Vec::new();
    walk(root, &mut HashSet::new(), &mut out);
    out
}

/// Child record ids of an encoded node, for topological repack, or `None` when
/// `data` is not a node of this key/value type.
#[must_use]
pub fn node_edges<K, V>(data: &[u8]) -> Option<Vec<[u8; 16]>>
where
    K: HamtCodec,
    V: HamtCodec,
{
    if data.get(..4) != Some(rezzy::hamt::HAMT_NODE_MAGIC.as_slice()) {
        return None;
    }
    let node = PersistedInternalNode::<K, V>::decode_v1_unverified(data).ok()?;
    Some(
        node.child_hashes
            .iter()
            .map(crate::state_hamt_node_id)
            .collect(),
    )
}

/// Stage encoded nodes into `collection` of `pool` within `txn`, so they
/// become visible in the same commit as the root records that name them.
///
/// Every node is written, including ones already stored. Content addressing
/// makes the bytes identical, and the write is what lets a garbage collector's
/// delete conflict with a publication that reuses an existing node.
///
/// # Errors
/// Returns the transaction's staging error.
pub fn put_nodes(
    txn: &DatabaseTransaction<'_>,
    pool: ShardType,
    collection: [u8; 16],
    nodes: &[(StructuralHash, Vec<u8>)],
) -> std::io::Result<()> {
    for (hash, bytes) in nodes {
        txn.put(
            pool,
            collection,
            crate::state_hamt_node_id(hash),
            &NodeData::new(bytes::Bytes::copy_from_slice(bytes)),
        )?;
    }
    Ok(())
}

/// One collection's HAMT nodes under one structural key.
pub struct HamtCollection<'a> {
    store: &'a PackfileStorage,
    collection: [u8; 16],
    structural_key: Vec<u8>,
}

impl<'a> HamtCollection<'a> {
    /// Nodes in `collection`, hashed under `structural_key`.
    #[must_use]
    pub fn new(
        store: &'a PackfileStorage,
        collection: [u8; 16],
        structural_key: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            store,
            collection,
            structural_key: structural_key.into(),
        }
    }

    /// The collection id.
    #[must_use]
    pub fn collection(&self) -> [u8; 16] {
        self.collection
    }

    /// The structural key mixed into every node hash.
    #[must_use]
    pub fn structural_key(&self) -> &[u8] {
        &self.structural_key
    }

    /// A verifying reader. Reuse it for related reads to share its node cache.
    #[must_use]
    pub fn reader<K, V>(&self) -> HamtReader<'_, K, V>
    where
        K: HamtCodec + Hash,
        V: HamtCodec + Hash,
    {
        let (store, collection) = (self.store, self.collection);
        let load: NodeLoader<'_> = Box::new(move |hash: &StructuralHash| {
            Ok(store
                .get(&collection, &crate::state_hamt_node_id(hash))?
                .map(|record| record.bytes.to_vec()))
        });
        PersistedHamtReader::new(&self.structural_key, load)
    }

    /// Look up `key` under `root`.
    ///
    /// # Errors
    /// Returns an error when a node on the path is missing, unreadable, or
    /// fails verification.
    pub fn get<K, V>(&self, root: StructuralHash, key: &K) -> Result<Option<V>, HamtStoreError>
    where
        K: HamtCodec + Hash + Eq,
        V: HamtCodec + Hash + Clone,
    {
        self.reader::<K, V>().get(root, key)
    }

    /// Every entry under `root`, in trie order.
    ///
    /// # Errors
    /// Returns an error when a node is missing, unreadable, or fails
    /// verification.
    pub fn collect_entries<K, V>(&self, root: StructuralHash) -> Result<Vec<(K, V)>, HamtStoreError>
    where
        K: HamtCodec + Hash + Clone,
        V: HamtCodec + Hash + Clone,
    {
        self.reader::<K, V>().collect_entries(root)
    }
}
