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

/// Per-object descriptor in the "objects" array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectDescriptor {
    #[serde(rename = "type")]
    pub obj_type: String,
    pub ndim: u64,
    pub shape: Vec<u64>,
    pub strides: Vec<u64>,
    pub dtype: Dtype,
    /// Arbitrary per-object metadata (namespaced keys, units, etc.)
    #[serde(flatten)]
    pub extra: BTreeMap<String, ciborium::Value>,
}

/// Per-object payload decoding instructions in the "payload" array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayloadDescriptor {
    pub byte_order: ByteOrder,
    pub encoding: String,
    pub filter: String,
    pub compression: String,
    /// Encoding/filter/compression parameters (reference_value, bits_per_value, etc.)
    /// and szip block offsets. All stored as ciborium::Value for flexibility.
    #[serde(flatten)]
    pub params: BTreeMap<String, ciborium::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<HashDescriptor>,
}

/// Top-level message metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub version: u64,
    pub objects: Vec<ObjectDescriptor>,
    pub payload: Vec<PayloadDescriptor>,
    /// All other top-level keys (namespaced metadata like "mars": {...})
    #[serde(flatten)]
    pub extra: BTreeMap<String, ciborium::Value>,
}
