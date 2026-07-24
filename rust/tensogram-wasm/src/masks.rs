// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! WASM binding for the advanced `decode_with_masks` API.
//!
//! Mirrors the Rust core [`tensogram::decode_with_masks`]: it returns
//! the `0.0`-substituted payload for every object **plus** the raw
//! decompressed NaN / +Inf / -Inf bitmasks, so advanced callers can
//! apply their own missing-value representation instead of the
//! canonical NaN / ±Inf restoration that [`crate::decode`] performs.
//!
//! # Mask representation
//!
//! Each mask is exposed to JavaScript as a `Uint8Array` of length
//! `n_elements` (the descriptor's shape product), one byte per tensor
//! element: `1` marks a position that held the corresponding
//! non-finite value at encode time, `0` marks a finite position.  A
//! mask kind that the frame did not carry is returned as `null`.  The
//! per-kind arrays never overlap at the same index (the encoder's
//! complex priority rule assigns each element to at most one kind — see
//! `plans/WIRE_FORMAT.md` §6.5.2).
//!
//! Mask arrays are JS-heap **copies** (via `Uint8Array::from`), so they
//! survive WASM linear-memory growth and outlive the owning
//! [`DecodedMaskedMessage`] handle — unlike the zero-copy payload views
//! returned by `object_data_u8`.

use crate::convert::*;
use tensogram::{self as core, DecodeOptions};
use wasm_bindgen::prelude::*;

/// Decode all objects from a message AND return the raw decompressed
/// NaN / Inf bitmasks alongside each `0.0`-substituted payload.
///
/// Mirrors [`tensogram::decode_with_masks`].  Unlike [`crate::decode`],
/// the payloads always contain `0.0` at non-finite positions —
/// restoration is intentionally **not** applied (the Rust core forces
/// `restore_non_finite = false` for this API).  Use the returned masks
/// to reconstruct or reinterpret the missing positions.
///
/// @param buf - Raw .tgm message bytes.
/// @param native_byte_order - When true (default), decoded payloads are
///                      converted to the host's native byte order
///                      regardless of the wire byte order declared in
///                      the descriptor.  Set to false to receive bytes
///                      in the message's declared wire byte order.
/// @returns A `DecodedMaskedMessage` handle.  Use `.object_data_u8(i)`
///          for the substituted payload and `.object_mask_nan(i)` /
///          `.object_mask_pos_inf(i)` / `.object_mask_neg_inf(i)` for
///          the per-kind masks (`Uint8Array` or `null`).
#[wasm_bindgen]
pub fn decode_with_masks(
    buf: &[u8],
    native_byte_order: Option<bool>,
) -> Result<DecodedMaskedMessage, JsValue> {
    let options = DecodeOptions {
        native_byte_order: native_byte_order.unwrap_or(true),
        // Stated explicitly for parity with the Rust API contract —
        // `decode_with_masks` forces this off regardless.
        restore_non_finite: false,
        ..Default::default()
    };
    let (metadata, objects) = core::decode_with_masks(buf, &options).map_err(js_err)?;
    let objects = objects
        .into_iter()
        .map(|o| MaskedObject {
            descriptor: o.descriptor,
            payload: o.payload,
            nan: o.masks.nan.map(bools_to_bytes),
            pos_inf: o.masks.pos_inf.map(bools_to_bytes),
            neg_inf: o.masks.neg_inf.map(bools_to_bytes),
        })
        .collect();
    Ok(DecodedMaskedMessage { metadata, objects })
}

/// Convert a decompressed `Vec<bool>` mask into the one-byte-per-element
/// `Vec<u8>` (`0` / `1`) representation exposed to JavaScript.
fn bools_to_bytes(bits: Vec<bool>) -> Vec<u8> {
    bits.into_iter().map(u8::from).collect()
}

/// One decoded object plus its per-kind masks, owned by a
/// [`DecodedMaskedMessage`].
struct MaskedObject {
    descriptor: core::DataObjectDescriptor,
    /// Decoded payload with `0.0` at every non-finite position.
    payload: Vec<u8>,
    /// One byte per element (`1` = NaN at encode time), or `None`.
    nan: Option<Vec<u8>>,
    /// One byte per element (`1` = `+∞`), or `None`.
    pos_inf: Option<Vec<u8>>,
    /// One byte per element (`1` = `−∞`), or `None`.
    neg_inf: Option<Vec<u8>>,
}

/// Handle to a message decoded via [`decode_with_masks`].
///
/// Owns the substituted payloads and the per-kind masks in WASM linear
/// memory.  The `object_data_u8` view is zero-copy (invalidated on WASM
/// memory growth — read or copy before the next WASM call); the mask
/// accessors return JS-heap copies that are always safe to retain.
///
/// Call `.free()` when done to release the WASM memory.
#[wasm_bindgen]
pub struct DecodedMaskedMessage {
    metadata: core::GlobalMetadata,
    objects: Vec<MaskedObject>,
}

#[wasm_bindgen]
impl DecodedMaskedMessage {
    /// Global metadata as a plain JS object (same shape as
    /// [`crate::DecodedMessage::metadata`]).
    pub fn metadata(&self) -> Result<JsValue, JsValue> {
        metadata_to_js(&self.metadata)
    }

    /// Number of data objects in the message.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Object descriptor (shape, dtype, encoding, masks sub-map, …) as
    /// a JS object.
    pub fn object_descriptor(&self, index: usize) -> Result<JsValue, JsValue> {
        to_js(&self.object(index)?.descriptor)
    }

    /// Zero-copy `Uint8Array` view into the `0.0`-substituted payload.
    ///
    /// **Warning**: this view points directly into WASM linear memory
    /// and is invalidated if WASM memory grows.  Read or copy the bytes
    /// before any further WASM calls.
    pub fn object_data_u8(&self, index: usize) -> Result<js_sys::Uint8Array, JsValue> {
        Ok(view_as_u8(&self.object(index)?.payload))
    }

    /// Raw payload byte length for the object at `index`.
    pub fn object_byte_length(&self, index: usize) -> Result<usize, JsValue> {
        Ok(self.object(index)?.payload.len())
    }

    /// NaN mask for the object at `index` as a `Uint8Array` (one byte
    /// per element, `1` = NaN), or `null` when the frame carried no
    /// NaN mask.  The array is a JS-heap copy — safe to retain.
    pub fn object_mask_nan(&self, index: usize) -> Result<JsValue, JsValue> {
        Ok(mask_to_js(self.object(index)?.nan.as_deref()))
    }

    /// `+∞` mask for the object at `index`, or `null`.  See
    /// [`Self::object_mask_nan`].
    pub fn object_mask_pos_inf(&self, index: usize) -> Result<JsValue, JsValue> {
        Ok(mask_to_js(self.object(index)?.pos_inf.as_deref()))
    }

    /// `−∞` mask for the object at `index`, or `null`.  See
    /// [`Self::object_mask_nan`].
    pub fn object_mask_neg_inf(&self, index: usize) -> Result<JsValue, JsValue> {
        Ok(mask_to_js(self.object(index)?.neg_inf.as_deref()))
    }
}

impl DecodedMaskedMessage {
    fn object(&self, index: usize) -> Result<&MaskedObject, JsValue> {
        self.objects.get(index).ok_or_else(|| {
            JsValue::from(js_sys::Error::new(&format!(
                "object index {index} out of range (have {})",
                self.objects.len()
            )))
        })
    }
}

/// Convert an optional mask byte-slice into a JS value: a JS-heap
/// `Uint8Array` copy when present, or `null` when the kind is absent.
fn mask_to_js(mask: Option<&[u8]>) -> JsValue {
    match mask {
        Some(bytes) => js_sys::Uint8Array::from(bytes).into(),
        None => JsValue::NULL,
    }
}
