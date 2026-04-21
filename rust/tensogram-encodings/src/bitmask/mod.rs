// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Compressed bitmasks for the NaN / Inf bitmask companion frame.
//!
//! Used by `NTensorFrame` (wire type 9, see
//! `plans/BITMASK_FRAME.md` §3) to record the positions of NaN / +Inf /
//! -Inf values in a float payload.  The payload itself has those
//! positions substituted with `0.0`; the bitmask tells the decoder
//! where to restore the non-finite values on decode.
//!
//! # Mask method registry
//!
//! Six methods, configurable per-mask via the [`MaskMethod`] enum:
//!
//! | Method | When to pick | Notes |
//! |---|---|---|
//! | [`MaskMethod::Roaring`] (default) | always | Hybrid array / bitmap / RLE containers.  Best all-rounder; [Roaring Portable Serialization Format](https://github.com/RoaringBitmap/RoaringFormatSpec). |
//! | [`MaskMethod::Rle`] | highly-clustered masks (land / sea masks, swath gaps) | Bandwidth-bound encode / decode; see [`rle`] for on-wire format. |
//! | [`MaskMethod::Blosc2`] | dense dtype-aligned masks | Reuses `BLOSC_BITSHUFFLE` filter + sub-codec; feature-gated. |
//! | [`MaskMethod::Zstd`] | generic good-ratio path | Reuses the main codec; always available. |
//! | [`MaskMethod::Lz4`] | decode-speed priority | Reuses the main codec. |
//! | [`MaskMethod::None`] | tiny masks (≤ small-mask threshold) | Raw packed bytes, no compression. |
//!
//! The small-mask fallback (auto-switch to `None` when the
//! uncompressed byte count is ≤ `small_mask_threshold_bytes`, default
//! 128) lives at the encoder-integration layer rather than here — this
//! module encodes whichever method the caller picks.

pub mod codecs;
pub mod packing;
pub mod rle;
pub mod roaring;

use thiserror::Error;

/// Bitmask compression method selector.  Serialised in the CBOR
/// descriptor's `masks[kind].method` field; see `plans/BITMASK_FRAME.md`
/// §3.3 for the schema.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum MaskMethod {
    /// Bit-level run-length encoding.  See [`rle`] for on-wire layout.
    ///
    /// Best on highly-clustered masks (land / sea, swath gaps).  Worst
    /// on random alternating bits — the small-mask fallback at the
    /// encoder-integration layer redirects to [`MaskMethod::None`] in
    /// that case.
    Rle,
    /// Roaring Bitmap, serialised via the standard
    /// [Roaring Portable Serialization Format](https://github.com/RoaringBitmap/RoaringFormatSpec).
    ///
    /// Default mask method on every platform (including `wasm32`).
    /// Hybrid containers (array / bitmap / RLE) adapt to mask density
    /// automatically.
    #[default]
    Roaring,
    /// Blosc2 with `BLOSC_BITSHUFFLE` filter and a sub-codec (default
    /// LZ4).  Feature-gated on `blosc2`.
    #[cfg(feature = "blosc2")]
    Blosc2 {
        /// Inner codec after the bit-shuffle.
        codec: crate::pipeline::Blosc2Codec,
        /// Codec-specific level (1..=9 for most).
        level: i32,
    },
    /// Zstandard on the bit-packed bytes.
    Zstd {
        /// Optional level; `None` uses the codec's default (3).
        level: Option<i32>,
    },
    /// LZ4 on the bit-packed bytes.
    Lz4,
    /// Uncompressed raw packed bytes — chosen automatically for tiny
    /// masks where compression overhead exceeds the savings.
    None,
}

impl MaskMethod {
    /// Canonical string name used in the CBOR `masks[kind].method`
    /// field.  Stable across the wire; do not change without a format
    /// version bump.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Rle => "rle",
            Self::Roaring => "roaring",
            #[cfg(feature = "blosc2")]
            Self::Blosc2 { .. } => "blosc2",
            Self::Zstd { .. } => "zstd",
            Self::Lz4 => "lz4",
            Self::None => "none",
        }
    }

    /// Inverse of [`name`].  Does not parse the `params` sub-map —
    /// method-specific parameters are handled by the encoder /
    /// decoder integration layer.  Returns [`MaskError::UnknownMethod`]
    /// if the name is not one of the known values.  Returns a
    /// sub-codec feature-disabled error for blosc2 if the feature is
    /// off at build time.
    pub fn from_name(name: &str) -> Result<Self, MaskError> {
        match name {
            "rle" => Ok(Self::Rle),
            "roaring" => Ok(Self::Roaring),
            "blosc2" => {
                #[cfg(feature = "blosc2")]
                {
                    Ok(Self::Blosc2 {
                        codec: crate::pipeline::Blosc2Codec::Lz4,
                        level: 5,
                    })
                }
                #[cfg(not(feature = "blosc2"))]
                {
                    Err(MaskError::FeatureDisabled { method: "blosc2" })
                }
            }
            "zstd" => Ok(Self::Zstd { level: None }),
            "lz4" => Ok(Self::Lz4),
            "none" => Ok(Self::None),
            other => Err(MaskError::UnknownMethod(other.to_string())),
        }
    }
}

/// Errors surfaced by the bitmask compress / decompress path.
///
/// Mapped to [`crate::TensogramError::Encoding`] /
/// [`crate::TensogramError::Compression`] by the encoder integration
/// layer.
#[derive(Debug, Error)]
pub enum MaskError {
    #[error(
        "unknown mask method {0:?} (expected \"none\" | \"rle\" | \"roaring\" | \"lz4\" | \"zstd\" | \"blosc2\")"
    )]
    UnknownMethod(String),
    #[error("mask method {method:?} requires feature {method:?} which is not compiled in")]
    FeatureDisabled { method: &'static str },
    #[error("bitmask length mismatch: expected {expected} elements, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },
    #[error("malformed mask payload: {0}")]
    Malformed(String),
    #[error("RLE decode error: {0}")]
    Rle(String),
    #[error("Roaring decode error: {0}")]
    Roaring(String),
    #[error("underlying codec error: {0}")]
    Codec(String),
}

/// A bitmask is represented in memory as a vector of `bool` with length
/// equal to the element count of the tensor it masks.  `true` at index
/// `i` means "position `i` holds the masked non-finite kind"; `false`
/// means "position `i` is finite (or a different kind of non-finite)".
///
/// Conversion to / from the bit-packed wire representation is handled
/// by [`packing`].
pub type Bitmask = Vec<bool>;
