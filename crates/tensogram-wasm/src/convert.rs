//! Conversion utilities between tensogram-core types and JS values.
//!
//! Uses `serde-wasm-bindgen` for metadata (CBOR → JS objects) and
//! manual `js_sys` typed array views for zero-copy tensor data.

use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Convert any serde-serializable value to a JsValue.
pub(crate) fn to_js<T: Serialize>(val: &T) -> Result<JsValue, JsError> {
    serde_wasm_bindgen::to_value(val).map_err(|e| JsError::new(&e.to_string()))
}

/// Create a zero-copy `Float32Array` view into a byte slice living in WASM memory.
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
    // Safety: `data` comes from Vec<u8> payloads returned by tensogram_core::decode().
    // On wasm32, the global allocator (dlmalloc) returns pointers aligned to ≥ 8 bytes
    // for any non-trivial allocation, satisfying f32's 4-byte alignment requirement.
    // The WASM spec also permits unaligned memory access, so this is safe in practice.
    // The view is valid for the lifetime of the caller's data; it points directly into
    // WASM linear memory and is invalidated if that memory grows.
    let f32_slice =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4) };
    Ok(unsafe { js_sys::Float32Array::view(f32_slice) })
}

/// Create a zero-copy `Float64Array` view.
///
/// # Safety
///
/// Same alignment rationale as [`view_as_f32`]: the allocator guarantees
/// ≥ 8-byte alignment for non-trivial allocations, satisfying f64's
/// 8-byte requirement.  The view is invalidated if WASM memory grows.
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
    let f64_slice =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, data.len() / 8) };
    Ok(unsafe { js_sys::Float64Array::view(f64_slice) })
}

/// Create a zero-copy `Int32Array` view.
///
/// # Safety
///
/// Same alignment rationale as [`view_as_f32`]: the allocator guarantees
/// ≥ 8-byte alignment, satisfying i32's 4-byte requirement.  The view is
/// invalidated if WASM memory grows.
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
    let i32_slice =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, data.len() / 4) };
    Ok(unsafe { js_sys::Int32Array::view(i32_slice) })
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
