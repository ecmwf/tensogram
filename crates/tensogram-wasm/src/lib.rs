// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! WebAssembly bindings for the Tensogram N-tensor message format.
//!
//! Provides encode, decode, scan, and streaming decode functions
//! accessible from JavaScript/TypeScript via wasm-bindgen.
//!
//! Tensor payloads are returned as zero-copy TypedArray views into
//! WASM linear memory for 60fps visualisation performance.
//!
//! # Compressor support
//!
//! This WASM build supports lz4, szip (pure-Rust), and zstd (pure-Rust).
//! Attempts to decode blosc2/zfp/sz3 compressed data will return an error.

mod convert;
mod streaming;

use convert::*;
use tensogram_core::{self as core, DecodeOptions, EncodeOptions};
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
#[wasm_bindgen]
pub fn decode(buf: &[u8], verify_hash: Option<bool>) -> Result<DecodedMessage, JsError> {
    let options = DecodeOptions {
        verify_hash: verify_hash.unwrap_or(false),
        ..Default::default()
    };
    let (metadata, objects) = core::decode(buf, &options).map_err(js_err)?;
    Ok(DecodedMessage { metadata, objects })
}

/// Decode only the global metadata from a message (no payload decoding).
///
/// @param buf - Raw .tgm message bytes
/// @returns Plain JS object with version, base, _reserved_, _extra_ fields
#[wasm_bindgen]
pub fn decode_metadata(buf: &[u8]) -> Result<JsValue, JsError> {
    let meta = core::decode_metadata(buf).map_err(js_err)?;
    to_js(&meta)
}

/// Decode a single object by index.
///
/// @param buf - Raw .tgm message bytes
/// @param index - Zero-based object index
/// @param verify_hash - Whether to verify hash
#[wasm_bindgen]
pub fn decode_object(
    buf: &[u8],
    index: usize,
    verify_hash: Option<bool>,
) -> Result<DecodedMessage, JsError> {
    let options = DecodeOptions {
        verify_hash: verify_hash.unwrap_or(false),
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
/// @returns Uint8Array containing the encoded .tgm message
#[wasm_bindgen]
pub fn encode(
    metadata_js: JsValue,
    objects_js: js_sys::Array,
    hash: Option<bool>,
) -> Result<js_sys::Uint8Array, JsError> {
    use core::hash::HashAlgorithm;
    use core::types::{DataObjectDescriptor, GlobalMetadata};

    let metadata: GlobalMetadata =
        serde_wasm_bindgen::from_value(metadata_js).map_err(|e| JsError::new(&e.to_string()))?;

    let mut descriptors = Vec::new();
    let mut data_vec: Vec<Vec<u8>> = Vec::new();

    for i in 0..objects_js.length() {
        let obj = objects_js.get(i);
        let desc_val = js_sys::Reflect::get(&obj, &"descriptor".into())
            .map_err(|_| JsError::new("each object must have a 'descriptor' field"))?;
        let data_val = js_sys::Reflect::get(&obj, &"data".into())
            .map_err(|_| JsError::new("each object must have a 'data' field"))?;

        let desc: DataObjectDescriptor =
            serde_wasm_bindgen::from_value(desc_val).map_err(|e| JsError::new(&e.to_string()))?;

        // Accept any TypedArray — view the underlying ArrayBuffer as raw bytes.
        let data_bytes = typed_array_to_bytes(&data_val).ok_or_else(|| {
            JsError::new(
                "data must be a TypedArray (Uint8Array, Float32Array, Float64Array, or Int32Array)",
            )
        })?;

        descriptors.push(desc);
        data_vec.push(data_bytes);
    }

    let options = EncodeOptions {
        hash_algorithm: if hash.unwrap_or(true) {
            Some(HashAlgorithm::Xxh3)
        } else {
            None
        },
        emit_preceders: false,
        ..Default::default()
    };

    let pairs: Vec<(&core::DataObjectDescriptor, &[u8])> = descriptors
        .iter()
        .zip(data_vec.iter())
        .map(|(d, v)| (d, v.as_slice()))
        .collect();
    let encoded = core::encode(&metadata, &pairs, &options).map_err(js_err)?;

    // Return a JS-owned copy.  We must not use view_as_u8 here because
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
    /// Global metadata as a plain JS object.
    pub fn metadata(&self) -> Result<JsValue, JsError> {
        to_js(&self.metadata)
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

// ── Internal helpers ─────────────────────────────────────────────────────────

fn js_err(e: core::TensogramError) -> JsError {
    JsError::new(&e.to_string())
}

/// Extract raw bytes from any supported TypedArray by viewing its ArrayBuffer.
///
/// Correctly respects `byteOffset` and `byteLength` so that subarrays /
/// views only yield the intended region (no data leak from the underlying
/// ArrayBuffer).
///
/// Returns `None` if `val` is not a recognised TypedArray type.
fn typed_array_to_bytes(val: &JsValue) -> Option<Vec<u8>> {
    if let Some(arr) = val.dyn_ref::<js_sys::Uint8Array>() {
        Some(arr.to_vec())
    } else if let Some(arr) = val.dyn_ref::<js_sys::Float32Array>() {
        Some(
            js_sys::Uint8Array::new_with_byte_offset_and_length(
                &arr.buffer(),
                arr.byte_offset(),
                arr.byte_length(),
            )
            .to_vec(),
        )
    } else if let Some(arr) = val.dyn_ref::<js_sys::Float64Array>() {
        Some(
            js_sys::Uint8Array::new_with_byte_offset_and_length(
                &arr.buffer(),
                arr.byte_offset(),
                arr.byte_length(),
            )
            .to_vec(),
        )
    } else if let Some(arr) = val.dyn_ref::<js_sys::Int32Array>() {
        Some(
            js_sys::Uint8Array::new_with_byte_offset_and_length(
                &arr.buffer(),
                arr.byte_offset(),
                arr.byte_length(),
            )
            .to_vec(),
        )
    } else {
        None
    }
}
