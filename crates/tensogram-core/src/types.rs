// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

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
/// The metadata frame CBOR has three named sections plus `version`:
/// - `base`: per-object metadata array — one entry per data object, each
///   entry holds ALL structured metadata for that object independently.
///   The encoder auto-populates `_reserved_.tensor` (ndim/shape/strides/dtype)
///   in each entry.
/// - `_reserved_`: library internals (provenance: encoder info, time, uuid).
///   Client code can read but MUST NOT write — the encoder validates this.
/// - `_extra_`: client-writable catch-all for ad-hoc message-level annotations.
///
/// Unknown CBOR keys at the top level are silently ignored on decode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalMetadata {
    pub version: u16,

    /// Per-object metadata array.  Each entry holds ALL structured metadata
    /// for that data object.  Entries are independent — no tracking of what
    /// is common across objects.
    ///
    /// The encoder auto-populates `_reserved_.tensor` (with ndim, shape,
    /// strides, dtype) in each entry.  Application code may pre-populate
    /// additional keys (e.g. `"mars": {…}`) before encoding; the encoder
    /// preserves them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub base: Vec<BTreeMap<String, ciborium::Value>>,

    /// Library internals — provenance info (encoder, time, uuid).
    /// Client code can read but MUST NOT write; the encoder overwrites this.
    #[serde(
        rename = "_reserved_",
        default,
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub reserved: BTreeMap<String, ciborium::Value>,

    /// Client-writable catch-all for ad-hoc message-level annotations.
    #[serde(
        rename = "_extra_",
        default,
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub extra: BTreeMap<String, ciborium::Value>,
}

/// Index frame payload — maps object ordinals to byte offsets.
#[derive(Debug, Clone, Default)]
pub struct IndexFrame {
    pub object_count: u64,
    /// Byte offset of each data object frame from message start.
    pub offsets: Vec<u64>,
    /// Total byte length of each data object frame, excluding alignment padding.
    pub lengths: Vec<u64>,
}

/// Hash frame payload — per-object integrity hashes.
#[derive(Debug, Clone)]
pub struct HashFrame {
    pub object_count: u64,
    pub hash_type: String,
    pub hashes: Vec<String>,
}

impl Default for GlobalMetadata {
    fn default() -> Self {
        Self {
            version: 2,
            base: Vec::new(),
            reserved: BTreeMap::new(),
            extra: BTreeMap::new(),
        }
    }
}

/// A decoded object: its descriptor paired with its raw decoded payload bytes.
pub type DecodedObject = (DataObjectDescriptor, Vec<u8>);
