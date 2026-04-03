use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::dtype::Dtype;

pub use tensogram_encodings::ByteOrder;

/// Hash descriptor for payload integrity verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashDescriptor {
    #[serde(rename = "type")]
    pub hash_type: String,
    pub value: String,
}

/// Per-object descriptor — merges tensor metadata and encoding instructions.
///
/// Each data object frame carries one of these as its CBOR descriptor.
/// This replaces the v1 split between `ObjectDescriptor` (tensor info)
/// and `PayloadDescriptor` (encoding info).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataObjectDescriptor {
    // ── Tensor metadata ──
    #[serde(rename = "type")]
    pub obj_type: String,
    pub ndim: u64,
    pub shape: Vec<u64>,
    pub strides: Vec<u64>,
    pub dtype: Dtype,

    // ── Encoding pipeline ──
    pub byte_order: ByteOrder,
    pub encoding: String,
    pub filter: String,
    pub compression: String,

    /// Encoding/filter/compression parameters (reference_value, bits_per_value,
    /// szip_block_offsets, etc.). Stored as ciborium::Value for flexibility.
    #[serde(flatten)]
    pub params: BTreeMap<String, ciborium::Value>,

    /// Per-object integrity hash (set during encoding).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<HashDescriptor>,
}

/// Global message metadata (carried in header/footer metadata frames).
///
/// Does NOT contain per-object information — that lives in each
/// data object frame's `DataObjectDescriptor`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalMetadata {
    pub version: u16,
    /// All other top-level keys (namespaced metadata like "mars": {...})
    #[serde(flatten)]
    pub extra: BTreeMap<String, ciborium::Value>,
}

/// Index frame payload — maps object ordinals to byte offsets.
#[derive(Debug, Clone, Default)]
pub struct IndexFrame {
    pub object_count: u64,
    /// Byte offset of each data object frame from message start.
    pub offsets: Vec<u64>,
    /// Payload length of each data object frame.
    pub lengths: Vec<u64>,
}

/// Hash frame payload — per-object integrity hashes.
#[derive(Debug, Clone)]
pub struct HashFrame {
    pub object_count: u64,
    pub hash_type: String,
    pub hashes: Vec<String>,
}

/// A decoded object: its descriptor paired with its raw decoded payload bytes.
pub type DecodedObject = (DataObjectDescriptor, Vec<u8>);
