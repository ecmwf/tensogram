// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Conversion utilities between tensogram types and JS values.
//!
//! Uses `serde-wasm-bindgen` for metadata (CBOR → JS objects) and
//! `wasm_bindgen::memory()` + byte offsets for zero-copy TypedArray
//! views that avoid forming misaligned Rust references.

use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Convert any serde-serializable value to a JsValue using a
/// JSON-compatible representation.
///
/// Rust `BTreeMap` / `HashMap` values serialise as plain JS objects
/// (not ES `Map`), which is the natural shape TypeScript consumers
/// expect and what the high-level `@ecmwf/tensogram` wrapper relies
/// on. Safe-range `u64` values come across as `number`; values that
/// exceed `Number.MAX_SAFE_INTEGER` come across as `BigInt`.
///
/// The existing `wasm-bindgen-test` suite round-trips through
/// `from_value`, which accepts both `Map` and plain object input,
/// so this switch is backwards-compatible.
pub(crate) fn to_js<T: Serialize>(val: &T) -> Result<JsValue, JsError> {
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    val.serialize(&serializer)
        .map_err(|e| JsError::new(&e.to_string()))
}

/// Create a zero-copy `Float32Array` view into a byte slice living in WASM memory.
///
/// Uses `wasm_bindgen::memory()` + byte offset to construct the view without
/// forming a `&[f32]` reference, avoiding any alignment UB.
///
/// # Safety
///
/// The returned view is only valid as long as the source data remains
/// alive and WASM linear memory is not grown (which would reallocate
/// the underlying `ArrayBuffer`).  The caller must read/copy the data
/// before any further WASM calls that might allocate.
pub(crate) fn view_as_f32(data: &[u8]) -> Result<js_sys::Float32Array, JsError> {
    if data.len() % 4 != 0 {
        return Err(JsError::new(&format!(
            "data length {} is not a multiple of 4 (Float32)",
            data.len()
        )));
    }
    if data.is_empty() {
        return Ok(js_sys::Float32Array::new_with_length(0));
    }
    let byte_offset = data.as_ptr() as u32;
    let length = (data.len() / 4) as u32;
    let memory = wasm_bindgen::memory().unchecked_into::<js_sys::WebAssembly::Memory>();
    Ok(js_sys::Float32Array::new_with_byte_offset_and_length(
        &memory.buffer(),
        byte_offset,
        length,
    ))
}

/// Create a zero-copy `Float64Array` view.
///
/// # Safety
///
/// Same rationale as [`view_as_f32`]: uses byte offsets into WASM memory,
/// no misaligned Rust references are formed.  Invalidated if WASM memory grows.
pub(crate) fn view_as_f64(data: &[u8]) -> Result<js_sys::Float64Array, JsError> {
    if data.len() % 8 != 0 {
        return Err(JsError::new(&format!(
            "data length {} is not a multiple of 8 (Float64)",
            data.len()
        )));
    }
    if data.is_empty() {
        return Ok(js_sys::Float64Array::new_with_length(0));
    }
    let byte_offset = data.as_ptr() as u32;
    let length = (data.len() / 8) as u32;
    let memory = wasm_bindgen::memory().unchecked_into::<js_sys::WebAssembly::Memory>();
    Ok(js_sys::Float64Array::new_with_byte_offset_and_length(
        &memory.buffer(),
        byte_offset,
        length,
    ))
}

/// Create a zero-copy `Int32Array` view.
///
/// # Safety
///
/// Same rationale as [`view_as_f32`]: uses byte offsets into WASM memory,
/// no misaligned Rust references are formed.  Invalidated if WASM memory grows.
pub(crate) fn view_as_i32(data: &[u8]) -> Result<js_sys::Int32Array, JsError> {
    if data.len() % 4 != 0 {
        return Err(JsError::new(&format!(
            "data length {} is not a multiple of 4 (Int32)",
            data.len()
        )));
    }
    if data.is_empty() {
        return Ok(js_sys::Int32Array::new_with_length(0));
    }
    let byte_offset = data.as_ptr() as u32;
    let length = (data.len() / 4) as u32;
    let memory = wasm_bindgen::memory().unchecked_into::<js_sys::WebAssembly::Memory>();
    Ok(js_sys::Int32Array::new_with_byte_offset_and_length(
        &memory.buffer(),
        byte_offset,
        length,
    ))
}

/// Create a zero-copy `Uint8Array` view.
pub(crate) fn view_as_u8(data: &[u8]) -> js_sys::Uint8Array {
    if data.is_empty() {
        return js_sys::Uint8Array::new_with_length(0);
    }
    unsafe { js_sys::Uint8Array::view(data) }
}

/// Create a safe-copy `Float32Array` (JS-heap owned).
pub(crate) fn copy_as_f32(data: &[u8]) -> Result<js_sys::Float32Array, JsError> {
    let view = view_as_f32(data)?;
    Ok(js_sys::Float32Array::new(&view))
}
