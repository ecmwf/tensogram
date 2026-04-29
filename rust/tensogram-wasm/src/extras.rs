// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Additional WASM exports covering the Scope C.1 API-parity gap.
//!
//! Each function here is a thin binding over a public function in the
//! Rust core or in `tensogram-encodings`.  All errors are mapped via
//! [`crate::convert::js_err`] so the TypeScript wrapper's
//! `mapTensogramError` sees a consistent message shape.

use crate::convert::*;
use serde::Serialize;
use tensogram::{self as core, DecodeOptions};
use tensogram_encodings::simple_packing;
use wasm_bindgen::prelude::*;

// ── decode_range ─────────────────────────────────────────────────────────────

/// Decode partial sub-tensor ranges from a single data object.
///
/// @param buf - Complete wire-format message bytes.
/// @param object_index - Zero-based index of the target object.
/// @param ranges - Flat `BigUint64Array` of `[offset0, count0, offset1, count1, …]`
///   pairs, in element units (not bytes).  Empty array returns an empty result.
///   hash is checked against the payload before any range is decoded.
/// @returns `{ descriptor, parts: Uint8Array[] }` — one raw-bytes view
///   per requested range, in request order.  Callers (e.g. the TS
///   wrapper's `decodeRange`) convert each `Uint8Array` into a
///   dtype-typed view.
///
/// Throws on out-of-range object index, unsupported `filter` (e.g.
/// shuffle), or bitmask dtype (matching the Rust-core contract).
///
/// Implementation note: the `parts` array is built manually with
/// `Uint8Array::from` rather than via `to_js` — `serde_wasm_bindgen`
/// serialises `&[u8]` as a plain JS `Array<number>` by default, and the
/// TS wrapper expects honest `Uint8Array` instances so that
/// `typedArrayFor` can wrap them zero-copy.
#[wasm_bindgen]
pub fn decode_range(
    buf: &[u8],
    object_index: usize,
    ranges: &js_sys::BigUint64Array,
) -> Result<JsValue, JsError> {
    let flat: Vec<u64> = ranges.to_vec();
    if !flat.len().is_multiple_of(2) {
        return Err(JsError::new(
            "ranges length must be a multiple of 2 (flat [offset, count] pairs)",
        ));
    }
    let range_pairs: Vec<(u64, u64)> = flat.chunks_exact(2).map(|w| (w[0], w[1])).collect();

    let options = DecodeOptions {
        ..Default::default()
    };
    let (descriptor, parts) =
        core::decode_range(buf, object_index, &range_pairs, &options).map_err(js_err)?;

    let result = js_sys::Object::new();
    js_sys::Reflect::set(&result, &"descriptor".into(), &to_js(&descriptor)?)
        .map_err(|_| JsError::new("failed to set descriptor"))?;
    // `js_sys::Array::new_with_length` is defined in terms of `u32`; a
    // Rust-side Vec of 2^32 entries would already have failed the
    // earlier `to_vec()` from the input `BigUint64Array` (WASM linear
    // memory is itself u32-indexed), so the cast here can't truncate
    // in any reachable execution.
    let parts_js = js_sys::Array::new_with_length(parts.len() as u32);
    for (i, bytes) in parts.iter().enumerate() {
        parts_js.set(i as u32, js_sys::Uint8Array::from(bytes.as_slice()).into());
    }
    js_sys::Reflect::set(&result, &"parts".into(), &parts_js)
        .map_err(|_| JsError::new("failed to set parts"))?;
    Ok(result.into())
}

// ── compute_hash ─────────────────────────────────────────────────────────────

/// Compute the hex-encoded hash of a byte slice.
///
/// @param data - Bytes to hash.
/// @param algo - Algorithm name; default `"xxh3"`.  Unknown algorithm
///   names raise a metadata error (matching `HashAlgorithm::parse`).
/// @returns The hex digest as a string (16 chars for xxh3-64).
#[wasm_bindgen]
pub fn compute_hash(data: &[u8], algo: Option<String>) -> Result<String, JsError> {
    let name = algo.as_deref().unwrap_or("xxh3");
    let algorithm = core::HashAlgorithm::parse(name).map_err(js_err)?;
    Ok(core::compute_hash(data, algorithm))
}

// ── simple_packing_compute_params ────────────────────────────────────────────

/// JS-side shape of the simple-packing params — matches the ``sp_``-prefixed
/// snake_case keys a descriptor's `params` map expects, so the caller can
/// spread the result straight into a descriptor literal.
#[derive(Serialize)]
struct SimplePackingParamsJs {
    sp_reference_value: f64,
    sp_binary_scale_factor: i32,
    sp_decimal_scale_factor: i32,
    sp_bits_per_value: u32,
}

/// Compute the simple-packing parameters (reference value, binary/decimal
/// scale factors, bits-per-value) for a float64 array.
///
/// @param values - `Float64Array` — finite, non-NaN.
/// @param bits_per_value - Quantization depth (0–64; 0 denotes a
///   constant-field packing).
/// @param decimal_scale_factor - Power-of-10 scaling applied before
///   packing.  Typically `0`.
/// @returns Plain JS object with ``sp_``-prefixed keys matching the
///   on-wire descriptor params: `{ sp_reference_value,
///   sp_binary_scale_factor, sp_decimal_scale_factor, sp_bits_per_value }`.
///   Spread into a descriptor to apply:
///   `{ ...computed, encoding: "simple_packing", …}`.
///
///   Note: the encoder also auto-computes these values when the
///   descriptor carries only `sp_bits_per_value` (and optionally
///   `sp_decimal_scale_factor`) — calling this function explicitly
///   is only needed if the caller wants to cache or inspect the
///   derived params across multiple encodes.
#[wasm_bindgen]
pub fn simple_packing_compute_params(
    values: &[f64],
    bits_per_value: u32,
    decimal_scale_factor: i32,
) -> Result<JsValue, JsError> {
    let params = simple_packing::compute_params(values, bits_per_value, decimal_scale_factor)
        .map_err(|e| JsError::new(&format!("encoding error: {e}")))?;
    to_js(&SimplePackingParamsJs {
        sp_reference_value: params.reference_value,
        sp_binary_scale_factor: params.binary_scale_factor,
        sp_decimal_scale_factor: params.decimal_scale_factor,
        sp_bits_per_value: params.bits_per_value,
    })
}

// ── encode_pre_encoded ───────────────────────────────────────────────────────

/// Encode a complete Tensogram message from pre-encoded data objects.
///
/// Like [`crate::encode`], but each object's bytes are assumed already
/// encoded by the caller (according to its descriptor's pipeline) and
/// are written verbatim.  The library validates descriptor structure and
/// any `szip_block_offsets` it finds but never runs the encoding
/// pipeline.  The hash is always recomputed from the caller's bytes.
///
/// @param metadata_js - GlobalMetadata (JS object, `version: 3` required).
/// @param objects_js - Array of `{descriptor, data}`; each `data` must
///   be a `Uint8Array` (opaque pre-encoded bytes).
/// @param hash - Whether to stamp an xxh3 hash.  Default `true`.
/// @returns Full wire-format message as `Uint8Array`.
#[wasm_bindgen]
pub fn encode_pre_encoded(
    metadata_js: JsValue,
    objects_js: js_sys::Array,
    hash: Option<bool>,
) -> Result<js_sys::Uint8Array, JsError> {
    let metadata = metadata_from_js(&metadata_js)?;
    let (descriptors, data_vec) = extract_descriptor_data_pairs(&objects_js)?;
    let pairs: Vec<(&core::DataObjectDescriptor, &[u8])> = descriptors
        .iter()
        .zip(data_vec.iter())
        .map(|(d, v)| (d, v.as_slice()))
        .collect();
    // encode_pre_encoded rejects strict-finite flags at the Rust
    // level; we hardcode them off here so the WASM surface can't
    // forward them accidentally.
    let encoded =
        core::encode_pre_encoded(&metadata, &pairs, &build_encode_options(hash)).map_err(js_err)?;
    Ok(js_sys::Uint8Array::from(encoded.as_slice()))
}

// ── validate_buffer ──────────────────────────────────────────────────────────

/// Validate a single Tensogram message buffer.  Returns a JSON string
/// matching the structure emitted by `tensogram::validate::ValidationReport`:
/// `{ issues: [...], object_count: N, hash_verified: bool }`.
///
/// The TypeScript wrapper parses this JSON once and exposes a typed
/// `ValidationReport`.  Keeping the bridge as a JSON string avoids
/// lossy conversion of large integers and keeps the WASM surface
/// language-neutral (it already matches the FFI contract).
///
/// @param buf - The wire-format bytes of a single message (not a file).
/// @param level - One of `"quick"` / `"default"` / `"checksum"` / `"full"`.
///   `None` defaults to `"default"`.
/// @param check_canonical - When true, adds RFC 8949 §4.2 canonical
///   CBOR key-ordering checks.
#[wasm_bindgen]
pub fn validate_buffer(
    buf: &[u8],
    level: Option<String>,
    check_canonical: bool,
) -> Result<String, JsError> {
    let options = parse_validate_options(level.as_deref(), check_canonical)?;
    let report = core::validate::validate_message(buf, &options);
    serde_json::to_string(&report).map_err(|e| JsError::new(&format!("encoding error: {e}")))
}

fn parse_validate_options(
    level: Option<&str>,
    check_canonical: bool,
) -> Result<core::validate::ValidateOptions, JsError> {
    use core::validate::{ValidateOptions, ValidationLevel};

    let (max_level, checksum_only) = match level.unwrap_or("default") {
        "quick" => (ValidationLevel::Structure, false),
        "default" => (ValidationLevel::Integrity, false),
        "checksum" => (ValidationLevel::Integrity, true),
        "full" => (ValidationLevel::Fidelity, false),
        other => {
            return Err(JsError::new(&format!(
                "unknown validation level '{other}', expected one of: quick, default, checksum, full",
            )));
        }
    };
    Ok(ValidateOptions {
        max_level,
        check_canonical,
        checksum_only,
    })
}
