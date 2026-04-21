// (C) Copyright 2026- ECMWF and individual contributors.
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
///
/// **Deprecated in v3.**  In v3 the per-object hash lives in the
/// inline hash slot of the data-object frame footer (see
/// `plans/WIRE_FORMAT.md` §2.2 and §2.4), not in the CBOR
/// descriptor.  This struct is retained only for the message-level
/// [`HashFrame`] CBOR schema (which stores an array of hex-encoded
/// digest values, to allow future longer digests).  Callers doing
/// frame-level integrity verification should go through
/// [`crate::hash::hash_frame_body`] /
/// [`crate::hash::verify_frame_hash`] instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashDescriptor {
    #[serde(rename = "type")]
    pub algorithm: String,
    pub value: String,
}

/// On-wire descriptor for one of the three NaN / Inf companion-frame
/// masks (see `plans/BITMASK_FRAME.md` §3.3).
///
/// `offset` and `length` locate the mask blob inside the frame's
/// payload region; `method` names the compression scheme (`rle`,
/// `roaring`, `blosc2`, `zstd`, `lz4`, or `none`); `params` carries
/// any method-specific parameters (e.g. zstd level, blosc2 sub-codec).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MaskDescriptor {
    /// Canonical method name — `rle` | `roaring` | `blosc2` | `zstd` |
    /// `lz4` | `none`.
    pub method: String,
    /// Byte offset of the mask blob, measured from the start of the
    /// frame's payload region (= the first byte after the 16-byte
    /// frame header).
    pub offset: u64,
    /// Byte length of the (compressed) mask blob on disk.
    pub length: u64,
    /// Method-specific parameters (e.g. `{ "level": 3 }` for zstd).
    /// Empty map is serialised as absent to match the canonical
    /// zero-cost form.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub params: BTreeMap<String, ciborium::Value>,
}

/// Top-level `masks` sub-map for the `NTensorFrame` (wire type 9,
/// see `plans/BITMASK_FRAME.md` §3.3).
///
/// All three fields are optional — a frame can carry any subset (or
/// none, in which case the entire `masks` sub-map is absent).  Field
/// names serialise as `nan`, `inf+`, `inf-` per the canonical sort
/// order (byte-lex: `inf+` < `inf-` < `nan`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MasksMetadata {
    /// Mask recording element positions that were NaN on encode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nan: Option<MaskDescriptor>,
    /// Mask recording element positions that were `+Inf` on encode.
    #[serde(rename = "inf+", default, skip_serializing_if = "Option::is_none")]
    pub pos_inf: Option<MaskDescriptor>,
    /// Mask recording element positions that were `-Inf` on encode.
    #[serde(rename = "inf-", default, skip_serializing_if = "Option::is_none")]
    pub neg_inf: Option<MaskDescriptor>,
}

impl MasksMetadata {
    /// `true` when every kind is absent.  In that case the `masks`
    /// field on the descriptor should be `None` rather than
    /// `Some(empty)`, to match the canonical zero-cost form.
    pub fn is_empty(&self) -> bool {
        self.nan.is_none() && self.pos_inf.is_none() && self.neg_inf.is_none()
    }
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

    /// Optional NaN / Inf companion-mask metadata (`NTensorFrame`,
    /// wire type 9 — see `plans/BITMASK_FRAME.md`).  `None` means no
    /// mask sections are present, and the frame is byte-compatible with
    /// the legacy `NTensorFrame` layout.
    ///
    /// Declared **before** `params` so that the flattened `params` map
    /// below does not absorb the `masks` key at deserialisation time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub masks: Option<MasksMetadata>,

    /// Encoding/filter/compression parameters (reference_value, bits_per_value,
    /// szip_block_offsets, etc.). Stored as ciborium::Value for flexibility.
    #[serde(flatten)]
    pub params: BTreeMap<String, ciborium::Value>,
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
///
/// v3 CBOR schema (see `plans/WIRE_FORMAT.md` §6.2):
///
/// ```cbor
/// { "offsets": [u64, ...], "lengths": [u64, ...] }
/// ```
///
/// Object count is derived from `offsets.len()`.  The previously
/// serialised `object_count` key is dropped.
#[derive(Debug, Clone, Default)]
pub struct IndexFrame {
    /// Byte offset of each data object frame from message start.
    pub offsets: Vec<u64>,
    /// Total byte length of each data object frame, excluding alignment padding.
    pub lengths: Vec<u64>,
}

/// Hash frame payload — per-object integrity hashes.
///
/// v3 CBOR schema (see `plans/WIRE_FORMAT.md` §6.3):
///
/// ```cbor
/// { "algorithm": "xxh3", "hashes": ["hex", "hex", ...] }
/// ```
///
/// The `hash_type` key was renamed to `algorithm` to signal that the
/// value names the algorithm rather than a type identifier.  Object
/// count is derived from `hashes.len()`.
#[derive(Debug, Clone)]
pub struct HashFrame {
    pub algorithm: String,
    pub hashes: Vec<String>,
}

impl Default for GlobalMetadata {
    fn default() -> Self {
        Self {
            version: crate::wire::WIRE_VERSION,
            base: Vec::new(),
            reserved: BTreeMap::new(),
            extra: BTreeMap::new(),
        }
    }
}

/// A decoded object: its descriptor paired with its raw decoded payload bytes.
pub type DecodedObject = (DataObjectDescriptor, Vec<u8>);
