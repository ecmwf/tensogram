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

// ── decode_descriptors ───────────────────────────────────────────────────────

/// Decode global metadata **and** per-object descriptors without
/// decoding any payload data.
///
/// Thin binding over [`tensogram::decode_descriptors`].  Cheaper than
/// [`crate::decode`] because the decompression / filter / endian-swap
/// pipeline never runs — only shapes, dtypes, and the metadata frame
/// are parsed.  Backs the lazy TypeScript `objects()` / `objectsMetadata()`
/// iterators.
///
/// @param buf - Raw .tgm message bytes.
/// @returns `{ metadata, descriptors }` — `metadata` is the same plain
///   JS object shape as [`crate::decode_metadata`], `descriptors` is an
///   array of descriptor objects (shape, dtype, encoding, …) in wire
///   order.
#[wasm_bindgen]
pub fn decode_descriptors(buf: &[u8]) -> Result<JsValue, JsValue> {
    let (metadata, descriptors) = core::decode_descriptors(buf).map_err(js_err)?;
    let result = js_sys::Object::new();
    set_field(&result, "metadata", &metadata_to_js(&metadata)?)?;
    set_field(&result, "descriptors", &to_js(&descriptors)?)?;
    Ok(result.into())
}

/// Set an own-property on a JS object, mapping the (practically
/// unreachable) `Reflect::set` failure to a thrown JS error.
fn set_field(obj: &js_sys::Object, key: &str, value: &JsValue) -> Result<(), JsValue> {
    js_sys::Reflect::set(obj, &JsValue::from_str(key), value)
        .map(|_| ())
        .map_err(|_| JsValue::from(js_sys::Error::new(&format!("failed to set '{key}'"))))
}

// ── scan_with_options ────────────────────────────────────────────────────────

/// Scan a multi-message buffer for message boundaries with explicit
/// options.
///
/// Thin binding over [`tensogram::scan_with_options`].  Returns an array
/// of `[offset, length]` pairs for each message found (identical shape
/// to [`crate::scan`]).
///
/// @param buf - Buffer potentially containing multiple .tgm messages.
/// @param bidirectional - Enable the meet-in-the-middle walker pair
///   (default `true`).  When `false`, walks forward from offset 0 only.
/// @param max_message_size - Upper bound (in bytes) on the `total_length`
///   a postamble may advertise; anything larger is treated as corruption
///   and the backward walker yields to the forward walker.  Omit for the
///   core default (4 GiB).
#[wasm_bindgen]
pub fn scan_with_options(
    buf: &[u8],
    bidirectional: Option<bool>,
    max_message_size: Option<f64>,
) -> Result<JsValue, JsValue> {
    let defaults = core::ScanOptions::default();
    let max_message_size = match max_message_size {
        Some(v) if v.is_finite() && v >= 0.0 => v as u64,
        Some(v) => {
            return Err(JsValue::from(js_sys::Error::new(&format!(
                "max_message_size must be a non-negative finite number, got {v}"
            ))));
        }
        None => defaults.max_message_size,
    };
    let opts = core::ScanOptions {
        bidirectional: bidirectional.unwrap_or(defaults.bidirectional),
        max_message_size,
    };
    let positions = core::scan_with_options(buf, &opts);
    to_js(&positions)
}

// ── data_object_inline_hashes ────────────────────────────────────────────────

/// Read the per-object inline hash slots from a wire-format message.
///
/// Thin binding over [`tensogram::data_object_inline_hashes`].  Walks
/// the frame chain of a **single** message and collects the inline hash
/// slot of every data-object frame, in emission order.
///
/// @param buf - Wire-format message bytes (a single message, not a file).
/// @returns An array with one entry per data object, in order: a
///   16-character lowercase xxh3-64 hex string (identical in form to
///   [`crate::compute_hash`]) when the frame carries a non-zero digest,
///   or `null` when the slot is empty (message encoded with hashing
///   disabled, or that frame opted out).
///
/// The hex-string representation matches the Python
/// `object_inline_hashes` and the C FFI `tgm_object_hash_value`
/// accessor, so digests are byte-for-byte comparable across bindings.
#[wasm_bindgen]
pub fn data_object_inline_hashes(buf: &[u8]) -> Result<JsValue, JsValue> {
    let hashes = core::data_object_inline_hashes(buf).map_err(js_err)?;
    let arr = js_sys::Array::new_with_length(hashes.len() as u32);
    for (i, h) in hashes.iter().enumerate() {
        let val = match h {
            Some(digest) => JsValue::from_str(&format!("{digest:016x}")),
            None => JsValue::NULL,
        };
        arr.set(i as u32, val);
    }
    Ok(arr.into())
}

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
) -> Result<JsValue, JsValue> {
    let flat: Vec<u64> = ranges.to_vec();
    if !flat.len().is_multiple_of(2) {
        return Err(JsValue::from(js_sys::Error::new(
            "ranges length must be a multiple of 2 (flat [offset, count] pairs)",
        )));
    }
    let range_pairs: Vec<(u64, u64)> = flat.chunks_exact(2).map(|w| (w[0], w[1])).collect();

    let options = DecodeOptions {
        ..Default::default()
    };
    let (descriptor, parts) =
        core::decode_range(buf, object_index, &range_pairs, &options).map_err(js_err)?;

    let result = js_sys::Object::new();
    js_sys::Reflect::set(&result, &"descriptor".into(), &to_js(&descriptor)?)
        .map_err(|_| JsValue::from(js_sys::Error::new("failed to set descriptor")))?;
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
        .map_err(|_| JsValue::from(js_sys::Error::new("failed to set parts")))?;
    Ok(result.into())
}

// ── compute_hash ─────────────────────────────────────────────────────────────

/// Compute the hex-encoded hash of a byte slice.
///
/// @param data - Bytes to hash.
/// @param algo - Algorithm name; default `"xxh3"`.  Unknown algorithm
///   names raise a metadata error.
/// @returns The hex digest as a string (16 chars for xxh3-64).
#[wasm_bindgen]
pub fn compute_hash(data: &[u8], algo: Option<String>) -> Result<String, JsValue> {
    let name = algo.as_deref().unwrap_or("xxh3");
    core::parse_hash_name(Some(name)).map_err(js_err)?;
    Ok(core::compute_hash(data))
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
) -> Result<JsValue, JsValue> {
    let params = simple_packing::compute_params(values, bits_per_value, decimal_scale_factor)
        .map_err(|e| JsValue::from(js_sys::Error::new(&format!("encoding error: {e}"))))?;
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
) -> Result<js_sys::Uint8Array, JsValue> {
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
) -> Result<String, JsValue> {
    let options = parse_validate_options(level.as_deref(), check_canonical)?;
    let report = core::validate::validate_message(buf, &options);
    serde_json::to_string(&report)
        .map_err(|e| JsValue::from(js_sys::Error::new(&format!("encoding error: {e}"))))
}

fn parse_validate_options(
    level: Option<&str>,
    check_canonical: bool,
) -> Result<core::validate::ValidateOptions, JsValue> {
    use core::validate::{ValidateOptions, ValidationLevel};

    let (max_level, checksum_only) = match level.unwrap_or("default") {
        "quick" => (ValidationLevel::Structure, false),
        "default" => (ValidationLevel::Integrity, false),
        "checksum" => (ValidationLevel::Integrity, true),
        "full" => (ValidationLevel::Fidelity, false),
        other => {
            return Err(JsValue::from(js_sys::Error::new(&format!(
                "unknown validation level '{other}', expected one of: quick, default, checksum, full",
            ))));
        }
    };
    Ok(ValidateOptions {
        max_level,
        check_canonical,
        checksum_only,
    })
}
