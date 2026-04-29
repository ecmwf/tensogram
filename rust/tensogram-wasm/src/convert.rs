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
//!
//! Also hosts the shared helpers (`js_err`, `build_encode_options`,
//! `extract_descriptor_data_pairs`) used by every WASM entrypoint
//! that writes a message.

use serde::Serialize;
use tensogram::{self as core, EncodeOptions};
use wasm_bindgen::prelude::*;

/// Convert any `Display`-able error to a `JsError` by stringifying it.
///
/// All WASM entrypoints funnel errors through this helper so the
/// TypeScript wrapper's `mapTensogramError` sees a consistent message
/// shape.
pub(crate) fn js_err<E: std::fmt::Display>(e: E) -> JsError {
    JsError::new(&e.to_string())
}

/// Build an `EncodeOptions` from the JS `hash: boolean` option.
///
/// Kept for call sites that don't yet pass the full kwargs (convert
/// shims, tests).  New entry points should call
/// [`build_encode_options_full`] directly.  Infallible — this form
/// cannot carry user-supplied mask-method names.
pub(crate) fn build_encode_options(hash: Option<bool>) -> EncodeOptions {
    build_encode_options_full(hash, None, None, None, None, None, None)
        .expect("build_encode_options_full with no method names cannot error")
}

/// Build an [`EncodeOptions`] from the full JS kwargs set for the
/// NaN / Inf bitmask companion frame.  See
/// `docs/src/guide/nan-inf-handling.md` for the semantics and
/// `plans/WIRE_FORMAT.md` §6.5 for the wire-format details.
///
/// - `hash`: `true` (default) or `false` to disable hashing.
/// - `allow_nan` / `allow_inf`: both default `false` (reject policy).
/// - `*_mask_method`: optional string name (`"none"` | `"rle"` |
///   `"roaring"` | `"lz4"` | `"zstd"` | `"blosc2"`).  Unknown names
///   return a `JsError` naming the offending value and the full
///   list of accepted names — no silent fallback.
/// - `small_mask_threshold_bytes`: default `128`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_encode_options_full(
    hash: Option<bool>,
    allow_nan: Option<bool>,
    allow_inf: Option<bool>,
    nan_mask_method: Option<&str>,
    pos_inf_mask_method: Option<&str>,
    neg_inf_mask_method: Option<&str>,
    small_mask_threshold_bytes: Option<usize>,
) -> Result<EncodeOptions, JsError> {
    use core::encode::MaskMethod;

    let defaults = EncodeOptions::default();
    let parse = |s: Option<&str>, d: MaskMethod| -> Result<MaskMethod, JsError> {
        match s {
            Some(name) => MaskMethod::from_name(name).map_err(js_err),
            None => Ok(d),
        }
    };
    Ok(EncodeOptions {
        hashing: hash.unwrap_or(true),
        allow_nan: allow_nan.unwrap_or(false),
        allow_inf: allow_inf.unwrap_or(false),
        nan_mask_method: parse(nan_mask_method, defaults.nan_mask_method.clone())?,
        pos_inf_mask_method: parse(pos_inf_mask_method, defaults.pos_inf_mask_method.clone())?,
        neg_inf_mask_method: parse(neg_inf_mask_method, defaults.neg_inf_mask_method.clone())?,
        small_mask_threshold_bytes: small_mask_threshold_bytes
            .unwrap_or(defaults.small_mask_threshold_bytes),
        ..defaults
    })
}

/// Pull a list of `{ descriptor, data }` pairs out of a JS array and
/// deserialise each descriptor via `serde_wasm_bindgen`.  Used by
/// `encode` and `encode_pre_encoded` which share the same input shape.
///
/// Returns `(descriptors, data_blobs)` — parallel vectors the caller
/// can `zip` into `&[(&DataObjectDescriptor, &[u8])]`.
pub(crate) fn extract_descriptor_data_pairs(
    objects_js: &js_sys::Array,
) -> Result<(Vec<core::DataObjectDescriptor>, Vec<Vec<u8>>), JsError> {
    let len = objects_js.length();
    let mut descriptors = Vec::with_capacity(len as usize);
    let mut data_vec = Vec::with_capacity(len as usize);
    for i in 0..len {
        let entry = objects_js.get(i);
        let desc_val = js_sys::Reflect::get(&entry, &"descriptor".into())
            .map_err(|_| JsError::new("each object must have a 'descriptor' field"))?;
        let data_val = js_sys::Reflect::get(&entry, &"data".into())
            .map_err(|_| JsError::new("each object must have a 'data' field"))?;
        let desc: core::DataObjectDescriptor =
            serde_wasm_bindgen::from_value(desc_val).map_err(js_err)?;
        let data_bytes = typed_array_or_u8_to_bytes(&data_val)
            .ok_or_else(|| JsError::new("data must be a TypedArray, DataView, or Uint8Array"))?;
        descriptors.push(desc);
        data_vec.push(data_bytes);
    }
    Ok((descriptors, data_vec))
}

/// Convert any serde-serializable value to a JsValue using a
/// JSON-compatible representation.
///
/// Rust `BTreeMap` / `HashMap` values serialise as plain JS objects
/// (not ES `Map`), which is the natural shape TypeScript consumers
/// expect and what the high-level `@ecmwf.int/tensogram` wrapper relies
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

/// Deserialize a JavaScript metadata object into a
/// [`tensogram::GlobalMetadata`] with the free-form routing rule.
///
/// Mirrors the Rust core's [`tensogram::metadata::cbor_to_global_metadata`]:
/// only `base`, `_reserved_`, and `_extra_` are library-interpreted.
/// Every other top-level key the caller supplied (for example a legacy
/// `"version"` or a free-form `"foo"`) flows into `_extra_`.  Explicit
/// `_extra_` entries beat free-form top-level keys on collision.
///
/// Using `serde_wasm_bindgen::from_value` directly would silently drop
/// unknown top-level fields, so the TypeScript binding would lose any
/// user-supplied free-form key at the WASM boundary — this helper
/// exists to close that gap.
pub(crate) fn metadata_from_js(
    metadata_js: &JsValue,
) -> Result<tensogram::GlobalMetadata, JsError> {
    use std::collections::BTreeMap;
    use tensogram::RESERVED_KEY;

    // Reject outright non-object input.
    if !metadata_js.is_object() || metadata_js.is_null() {
        return Err(JsError::new(&format!(
            "metadata must be a plain object, got {metadata_js:?}"
        )));
    }

    let obj: &js_sys::Object = metadata_js.unchecked_ref();

    // Reject `_reserved_` at the top level — library-managed namespace.
    if js_sys::Reflect::has(obj, &JsValue::from_str(RESERVED_KEY))
        .map_err(|_| JsError::new("internal: Reflect.has failed on metadata object"))?
    {
        return Err(JsError::new(&format!(
            "'{RESERVED_KEY}' must not be set by client code — the encoder populates it"
        )));
    }

    // Pull `base` explicitly so serde-wasm-bindgen handles the nested
    // CBOR conversion (lists of dicts → Vec<BTreeMap>).  Absent ≡ empty.
    let base: Vec<BTreeMap<String, ciborium::Value>> =
        match js_sys::Reflect::get(obj, &JsValue::from_str("base"))
            .map_err(|_| JsError::new("internal: Reflect.get('base') failed"))?
        {
            v if v.is_undefined() => Vec::new(),
            v => serde_wasm_bindgen::from_value(v).map_err(js_err)?,
        };

    // Reject `_reserved_` inside any `base[i]` entry for parity with
    // the Rust core + Python validators.
    for (i, entry) in base.iter().enumerate() {
        if entry.contains_key(RESERVED_KEY) {
            return Err(JsError::new(&format!(
                "base[{i}] must not contain '{RESERVED_KEY}' — the encoder populates it"
            )));
        }
    }

    // Pull `_extra_` explicitly (authoritative — wins collisions).
    let mut extra: BTreeMap<String, ciborium::Value> =
        match js_sys::Reflect::get(obj, &JsValue::from_str("_extra_"))
            .map_err(|_| JsError::new("internal: Reflect.get('_extra_') failed"))?
        {
            v if v.is_undefined() => BTreeMap::new(),
            v => serde_wasm_bindgen::from_value(v).map_err(js_err)?,
        };

    // Everything else becomes a free-form `_extra_` entry.  Explicit
    // `_extra_` beats implicit free-form on same-name collisions.
    const KNOWN: &[&str] = &["base", "_extra_", RESERVED_KEY];
    let own_keys = js_sys::Object::keys(obj);
    for i in 0..own_keys.length() {
        let key_val = own_keys.get(i);
        let key = match key_val.as_string() {
            Some(s) => s,
            None => continue, // defensive: non-string key
        };
        if KNOWN.contains(&key.as_str()) {
            continue;
        }
        if extra.contains_key(&key) {
            continue; // explicit _extra_ already claimed this slot
        }
        let value = js_sys::Reflect::get(obj, &key_val)
            .map_err(|_| JsError::new("internal: Reflect.get for free-form key failed"))?;
        let cbor: ciborium::Value = serde_wasm_bindgen::from_value(value).map_err(js_err)?;
        extra.insert(key, cbor);
    }

    Ok(tensogram::GlobalMetadata {
        base,
        reserved: BTreeMap::new(),
        extra,
    })
}

/// Serialize a [`GlobalMetadata`] to JavaScript with the wire-format
/// version injected as `version`.
///
/// The `version` field was removed from the Rust `GlobalMetadata`
/// struct in v3 — the wire version now lives exclusively in the
/// preamble (see `plans/WIRE_FORMAT.md` §3).  TypeScript callers
/// still observe a `version` key on decoded metadata for ergonomic
/// access; this helper synthesises it from [`tensogram::WIRE_VERSION`].
///
/// Any user-supplied free-form `"version"` key in `_extra_` remains
/// reachable via `metadata._extra_.version` — the synthetic top-level
/// `version` is purely the wire-format answer.
pub(crate) fn metadata_to_js(meta: &tensogram::GlobalMetadata) -> Result<JsValue, JsError> {
    let obj = to_js(meta)?;
    // The serialised form is a plain JS object (via json_compatible
    // serializer).  Inject the synthetic `version` field in place so
    // callers always see it, regardless of whether the CBOR frame
    // carried one (it does not, in v3).
    let reflect = js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("version"),
        &JsValue::from(tensogram::WIRE_VERSION),
    );
    if reflect.is_err() {
        return Err(JsError::new(
            "internal: failed to set synthetic `version` on metadata JS object",
        ));
    }
    Ok(obj)
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
    if !data.len().is_multiple_of(4) {
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
    if !data.len().is_multiple_of(8) {
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
    if !data.len().is_multiple_of(4) {
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

/// Extract raw bytes from any TypedArray, DataView, or Uint8Array-like.
///
/// Used by write paths that accept "raw payload" input — encode,
/// encode_pre_encoded, StreamingEncoder::write_object etc.  Accepts
/// Uint8Array, Uint8ClampedArray, Int8Array, any of the (u)int/float
/// 16/32/64 typed arrays, plus DataView.  Returns `None` for anything
/// else so callers can emit a clear error.
///
/// The bytes are always copied into a fresh Vec so the caller owns
/// them — avoids lifetime tangles with the JS-side ArrayBuffer.
pub(crate) fn typed_array_or_u8_to_bytes(val: &JsValue) -> Option<Vec<u8>> {
    // Fast path: Uint8Array directly.
    if let Some(arr) = val.dyn_ref::<js_sys::Uint8Array>() {
        return Some(arr.to_vec());
    }
    // DataView: take byteOffset + byteLength view into the buffer.
    if let Some(dv) = val.dyn_ref::<js_sys::DataView>() {
        let u8 = js_sys::Uint8Array::new_with_byte_offset_and_length(
            &dv.buffer(),
            dv.byte_offset() as u32,
            dv.byte_length() as u32,
        );
        return Some(u8.to_vec());
    }
    // Every other TypedArray: inspect via the ArrayBufferView prototype.
    // js_sys exposes `byte_offset()` / `byte_length()` / `buffer()` on
    // each concrete class; we try them in turn.
    macro_rules! try_typed {
        ($ty:ty) => {
            if let Some(arr) = val.dyn_ref::<$ty>() {
                let u8 = js_sys::Uint8Array::new_with_byte_offset_and_length(
                    &arr.buffer(),
                    arr.byte_offset(),
                    arr.byte_length(),
                );
                return Some(u8.to_vec());
            }
        };
    }
    try_typed!(js_sys::Int8Array);
    try_typed!(js_sys::Uint8ClampedArray);
    try_typed!(js_sys::Int16Array);
    try_typed!(js_sys::Uint16Array);
    try_typed!(js_sys::Int32Array);
    try_typed!(js_sys::Uint32Array);
    try_typed!(js_sys::Float32Array);
    try_typed!(js_sys::Float64Array);
    try_typed!(js_sys::BigInt64Array);
    try_typed!(js_sys::BigUint64Array);
    None
}
