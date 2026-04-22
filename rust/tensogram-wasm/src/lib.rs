// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! WebAssembly bindings for the Tensogram N-tensor message format.
//!
//! Provides encode, decode, scan, streaming decode, range decode, hash,
//! validation, pre-encoded encode, `simple_packing` params, and a
//! frame-at-a-time `StreamingEncoder` — accessible from JavaScript /
//! TypeScript via `wasm-bindgen`.
//!
//! Tensor payloads are returned as zero-copy TypedArray views into
//! WASM linear memory for 60fps visualisation performance.
//!
//! # Compressor support
//!
//! This WASM build supports lz4, szip (pure-Rust), and zstd (pure-Rust).
//! Attempts to decode blosc2/zfp/sz3 compressed data will return an error.

mod convert;
mod encoder;
mod extras;
mod streaming;

use convert::*;
use tensogram::{self as core, DecodeOptions};
use wasm_bindgen::prelude::*;

// ── Decode API ───────────────────────────────────────────────────────────────

/// Decode all objects from a complete Tensogram message.
///
/// Returns a `DecodedMessage` handle that owns the decoded data.
/// Use `.object_data_f32(i)` etc. to get zero-copy TypedArray views
/// into the decoded payloads.
///
/// @param buf - Raw .tgm message bytes
/// @param verify_hash - Whether to verify payload integrity hashes (default: false)
/// @param restore_non_finite - When true (default), decode writes canonical
///                             NaN / +Inf / -Inf at positions recorded in
///                             the frame's mask companion.  Set to false to
///                             receive 0.0-substituted bytes as on disk.
#[wasm_bindgen]
pub fn decode(
    buf: &[u8],
    verify_hash: Option<bool>,
    restore_non_finite: Option<bool>,
) -> Result<DecodedMessage, JsError> {
    let options = DecodeOptions {
        verify_hash: verify_hash.unwrap_or(false),
        restore_non_finite: restore_non_finite.unwrap_or(true),
        ..Default::default()
    };
    let (metadata, objects) = core::decode(buf, &options).map_err(js_err)?;
    Ok(DecodedMessage { metadata, objects })
}

/// Decode only the global metadata from a message (no payload decoding).
///
/// @param buf - Raw .tgm message bytes
/// @returns Plain JS object with version (synthesised from the
///   preamble), base, _reserved_, _extra_ fields
#[wasm_bindgen]
pub fn decode_metadata(buf: &[u8]) -> Result<JsValue, JsError> {
    let meta = core::decode_metadata(buf).map_err(js_err)?;
    metadata_to_js(&meta)
}

/// Decode a single object by index.
///
/// @param buf - Raw .tgm message bytes
/// @param index - Zero-based object index
/// @param verify_hash - Whether to verify hash
/// @param restore_non_finite - Restore canonical NaN / Inf from mask companion (default: true)
#[wasm_bindgen]
pub fn decode_object(
    buf: &[u8],
    index: usize,
    verify_hash: Option<bool>,
    restore_non_finite: Option<bool>,
) -> Result<DecodedMessage, JsError> {
    let options = DecodeOptions {
        verify_hash: verify_hash.unwrap_or(false),
        restore_non_finite: restore_non_finite.unwrap_or(true),
        ..Default::default()
    };
    let (metadata, descriptor, data) = core::decode_object(buf, index, &options).map_err(js_err)?;
    Ok(DecodedMessage {
        metadata,
        objects: vec![(descriptor, data)],
    })
}

/// Scan a buffer for concatenated Tensogram messages.
///
/// Returns an array of `[offset, length]` pairs for each message found.
///
/// @param buf - Buffer potentially containing multiple .tgm messages
/// @returns Array of [offset, length] pairs
#[wasm_bindgen]
pub fn scan(buf: &[u8]) -> Result<JsValue, JsError> {
    let positions = core::scan(buf);
    to_js(&positions)
}

// ── Encode API ───────────────────────────────────────────────────────────────

/// Encode objects into a Tensogram message.
///
/// @param metadata_js - GlobalMetadata as a plain JS object
/// @param objects_js - Array of {descriptor, data} objects where data is a TypedArray
/// @param hash - Whether to compute integrity hashes (default: true)
/// @param allow_nan - When true, substitute NaN with 0 and record
///                    positions in a mask companion frame (default: false)
/// @param allow_inf - When true, substitute +Inf / -Inf with 0 and
///                    record positions in per-sign masks (default: false)
/// @param nan_mask_method - Mask compression method for the NaN mask
/// @param pos_inf_mask_method - Mask compression method for the +Inf mask
/// @param neg_inf_mask_method - Mask compression method for the -Inf mask
/// @param small_mask_threshold_bytes - Mask size below which method="none" is forced (default: 128)
/// @returns Uint8Array containing the encoded .tgm message
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn encode(
    metadata_js: JsValue,
    objects_js: js_sys::Array,
    hash: Option<bool>,
    allow_nan: Option<bool>,
    allow_inf: Option<bool>,
    nan_mask_method: Option<String>,
    pos_inf_mask_method: Option<String>,
    neg_inf_mask_method: Option<String>,
    small_mask_threshold_bytes: Option<usize>,
) -> Result<js_sys::Uint8Array, JsError> {
    let metadata: core::GlobalMetadata =
        serde_wasm_bindgen::from_value(metadata_js).map_err(js_err)?;
    let (descriptors, data_vec) = extract_descriptor_data_pairs(&objects_js)?;
    let pairs: Vec<(&core::DataObjectDescriptor, &[u8])> = descriptors
        .iter()
        .zip(data_vec.iter())
        .map(|(d, v)| (d, v.as_slice()))
        .collect();
    let options = build_encode_options_full(
        hash,
        allow_nan,
        allow_inf,
        nan_mask_method.as_deref(),
        pos_inf_mask_method.as_deref(),
        neg_inf_mask_method.as_deref(),
        small_mask_threshold_bytes,
    )?;
    let encoded = core::encode(&metadata, &pairs, &options).map_err(js_err)?;
    // Return a JS-owned copy.  We must not use `view_as_u8` here because
    // `encoded` is a local Vec that will be dropped when this function
    // returns — a view into it would be a dangling pointer.
    Ok(js_sys::Uint8Array::from(encoded.as_slice()))
}

// ── DecodedMessage handle ────────────────────────────────────────────────────

/// Handle to a decoded Tensogram message.
///
/// Owns the decoded payload data in WASM linear memory.  Use the
/// `object_data_*` methods to get zero-copy TypedArray views.
///
/// **Important**: The returned TypedArray views are invalidated if WASM
/// memory grows.  Read or copy the data before further WASM calls.
/// Call `.free()` when done to release WASM memory.
#[wasm_bindgen]
pub struct DecodedMessage {
    metadata: core::GlobalMetadata,
    objects: Vec<core::DecodedObject>,
}

#[wasm_bindgen]
impl DecodedMessage {
    /// Global metadata as a plain JS object.  The wire-format
    /// `version` is synthesised from the preamble (v3: always `3`)
    /// for TypeScript ergonomics — see `metadata_to_js` in
    /// `convert.rs`.
    pub fn metadata(&self) -> Result<JsValue, JsError> {
        metadata_to_js(&self.metadata)
    }

    /// Number of data objects in the message.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Object descriptor (shape, dtype, encoding, etc.) as a JS object.
    pub fn object_descriptor(&self, index: usize) -> Result<JsValue, JsError> {
        // Reuse payload() for the bounds check so the error message is consistent.
        let _ = self.payload(index)?;
        to_js(&self.objects[index].0)
    }

    // ── Zero-copy TypedArray views ───────────────────────────────────────

    /// Zero-copy Float32Array view into the decoded payload.
    ///
    /// **Warning**: This view points directly into WASM linear memory.
    /// It becomes invalid if WASM memory grows.  Read the data or pass
    /// it to WebGL before any further WASM calls.
    pub fn object_data_f32(&self, index: usize) -> Result<js_sys::Float32Array, JsError> {
        let data = self.payload(index)?;
        view_as_f32(data)
    }

    /// Zero-copy Float64Array view.
    pub fn object_data_f64(&self, index: usize) -> Result<js_sys::Float64Array, JsError> {
        let data = self.payload(index)?;
        view_as_f64(data)
    }

    /// Zero-copy Int32Array view.
    pub fn object_data_i32(&self, index: usize) -> Result<js_sys::Int32Array, JsError> {
        let data = self.payload(index)?;
        view_as_i32(data)
    }

    /// Zero-copy Uint8Array view.
    pub fn object_data_u8(&self, index: usize) -> Result<js_sys::Uint8Array, JsError> {
        let data = self.payload(index)?;
        Ok(view_as_u8(data))
    }

    // ── Safe-copy variants ───────────────────────────────────────────────

    /// Safe-copy Float32Array (JS-heap owned, survives WASM memory growth).
    pub fn object_data_copy_f32(&self, index: usize) -> Result<js_sys::Float32Array, JsError> {
        let data = self.payload(index)?;
        copy_as_f32(data)
    }

    /// Raw payload byte length for object at `index`.
    pub fn object_byte_length(&self, index: usize) -> Result<usize, JsError> {
        Ok(self.payload(index)?.len())
    }
}

impl DecodedMessage {
    fn payload(&self, index: usize) -> Result<&[u8], JsError> {
        if index >= self.objects.len() {
            return Err(JsError::new(&format!(
                "object index {index} out of range (have {})",
                self.objects.len()
            )));
        }
        Ok(&self.objects[index].1)
    }
}

// ── StreamingDecoder re-export ───────────────────────────────────────────────

pub use streaming::StreamingDecoder;

// ── StreamingEncoder re-export ───────────────────────────────────────────────

pub use encoder::StreamingEncoder;

// ── Scope-C exports (decode_range, compute_hash, validate, …) ───────────────

pub use extras::{
    compute_hash, decode_range, encode_pre_encoded, simple_packing_compute_params, validate_buffer,
};
