// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

// FFI functions accept raw pointers by design — callers are responsible for
// validity. Marking every extern "C" fn as `unsafe` would be correct but
// makes cbindgen emit ugly signatures with no benefit to C callers.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

//! Tensogram C FFI
//!
//! Exposes the tensogram library to C and C++ callers via opaque handles,
//! typed accessor functions, and a flat C ABI.
//!
//! Memory ownership rules:
//! - Handles returned by `tgm_*` functions are owned by the caller.
//!   Free them with the matching `tgm_*_free` function.
//! - Pointers returned by accessor functions (e.g. `tgm_object_shape`) are
//!   borrowed from the handle and valid until the handle is freed.
//! - `tgm_bytes_t` returned by encode functions must be freed with `tgm_bytes_free`.
//!
//! ## JSON schema for `tgm_encode`
//!
//! The `metadata_json` argument to `tgm_encode` must be a JSON object with:
//! - `"version"` (integer, required): wire format version (3).
//! - `"descriptors"` (array, required): one entry per data object. Each entry
//!   merges tensor info and encoding pipeline info into a single object:
//!   `type`, `ndim`, `shape`, `strides`, `dtype`, `byte_order`, `encoding`,
//!   `filter`, `compression`. Additional keys are stored as params.
//! - Any other top-level keys (e.g. `"mars"`) are stored as global extra metadata.

use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use std::ptr;
use std::slice;

use tensogram::encode::MaskMethod;
use tensogram::validate::{
    ValidateOptions, ValidationLevel, validate_file as core_validate_file, validate_message,
};
use tensogram::{
    DataObjectDescriptor, DecodeOptions, EncodeOptions, GlobalMetadata, HashAlgorithm,
    RESERVED_KEY, StreamingEncoder, TensogramError, TensogramFile, decode, decode_metadata,
    decode_object, decode_range, encode, encode_pre_encoded, scan,
};

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------

#[repr(C)]
pub enum TgmError {
    Ok = 0,
    Framing = 1,
    Metadata = 2,
    Encoding = 3,
    Compression = 4,
    Object = 5,
    Io = 6,
    HashMismatch = 7,
    InvalidArg = 8,
    /// Returned by `tgm_*_iter_next` when iteration is exhausted.
    EndOfIter = 9,
    Remote = 10,
}

fn to_error_code(e: &TensogramError) -> TgmError {
    match e {
        TensogramError::Framing(_) => TgmError::Framing,
        TensogramError::Metadata(_) => TgmError::Metadata,
        TensogramError::Encoding(_) => TgmError::Encoding,
        TensogramError::Compression(_) => TgmError::Compression,
        TensogramError::Object(_) => TgmError::Object,
        TensogramError::Io(_) => TgmError::Io,
        TensogramError::HashMismatch { .. } => TgmError::HashMismatch,
        TensogramError::Remote(_) => TgmError::Remote,
    }
}

// Thread-local storage for the last error message.
thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<CString>> = const { std::cell::RefCell::new(None) };
}

fn set_last_error(msg: &str) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(msg).ok();
    });
}

/// Returns a pointer to the last error message, or NULL if no error.
/// The pointer is valid until the next FFI call on the same thread.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    })
}

// ---------------------------------------------------------------------------
// Byte buffer
// ---------------------------------------------------------------------------

/// An owned byte buffer returned by encode functions.
#[repr(C)]
pub struct TgmBytes {
    pub data: *mut u8,
    pub len: usize,
}

/// Mask-companion options for encode entry points (see
/// `plans/BITMASK_FRAME.md` §6.3).  Pass a pointer to this struct to
/// opt into NaN / ±Inf substitution with bitmask companion frames.
///
/// Each `*_mask_method` string is one of `"none"`, `"rle"`,
/// `"roaring"`, `"lz4"`, `"zstd"`, or `"blosc2"`; pass `NULL` to use
/// the library default (`"roaring"`).  Unknown names cause the
/// owning `tgm_*_with_options` call to return
/// [`TgmError::InvalidArg`] with a clear message via
/// [`tgm_last_error`].
///
/// `small_mask_threshold_bytes` is the byte-count below which mask
/// blobs are written as `"none"` regardless of the requested method
/// (auto-fallback).  Pass `0` to disable the fallback.  Negative
/// values use the library default (128).
#[repr(C)]
pub struct TgmEncodeMaskOptions {
    pub allow_nan: bool,
    pub allow_inf: bool,
    pub nan_mask_method: *const c_char,
    pub pos_inf_mask_method: *const c_char,
    pub neg_inf_mask_method: *const c_char,
    pub small_mask_threshold_bytes: isize,
}

/// Parse one of the optional C-string mask-method fields into a Rust
/// [`MaskMethod`].  Returns the caller-supplied default on `NULL`,
/// an `Err` naming the offending value (and accepted alternatives)
/// for invalid UTF-8 or unknown names.
///
/// # Safety
///
/// `ptr` must either be `NULL` or point to a NUL-terminated UTF-8
/// string with a valid Rust-bound lifetime.
unsafe fn parse_mask_method_cstr(
    ptr: *const c_char,
    default: MaskMethod,
) -> Result<MaskMethod, String> {
    if ptr.is_null() {
        return Ok(default);
    }
    let s = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|_| "mask method name is not valid UTF-8".to_string())?;
    MaskMethod::from_name(s).map_err(|e| e.to_string())
}

/// Apply the optional [`TgmEncodeMaskOptions`] pointer to an
/// [`EncodeOptions`].  `NULL` is a no-op.  Returns an error message
/// (routed to [`set_last_error`] by the caller) when a method name
/// is invalid UTF-8 or unknown.
///
/// # Safety
///
/// `opts` must either be `NULL` or point to a valid
/// `TgmEncodeMaskOptions` whose `*_mask_method` fields satisfy
/// [`parse_mask_method_cstr`]'s safety contract.
unsafe fn apply_mask_options(
    encode_opts: &mut EncodeOptions,
    opts: *const TgmEncodeMaskOptions,
) -> Result<(), String> {
    if opts.is_null() {
        return Ok(());
    }
    let opts = unsafe { &*opts };
    encode_opts.allow_nan = opts.allow_nan;
    encode_opts.allow_inf = opts.allow_inf;
    encode_opts.nan_mask_method =
        unsafe { parse_mask_method_cstr(opts.nan_mask_method, MaskMethod::default())? };
    encode_opts.pos_inf_mask_method =
        unsafe { parse_mask_method_cstr(opts.pos_inf_mask_method, MaskMethod::default())? };
    encode_opts.neg_inf_mask_method =
        unsafe { parse_mask_method_cstr(opts.neg_inf_mask_method, MaskMethod::default())? };
    if opts.small_mask_threshold_bytes >= 0 {
        encode_opts.small_mask_threshold_bytes = opts.small_mask_threshold_bytes as usize;
    }
    Ok(())
}

/// Decode-side companion to [`TgmEncodeMaskOptions`].  Pass a pointer
/// to opt out of canonical NaN / Inf restoration.  Pass `NULL` for
/// the default `restore_non_finite = true`.
#[repr(C)]
pub struct TgmDecodeMaskOptions {
    pub restore_non_finite: bool,
}

/// Apply the optional [`TgmDecodeMaskOptions`] pointer to a
/// [`DecodeOptions`].  `NULL` is a no-op.
///
/// # Safety
///
/// `opts` must either be `NULL` or point to a valid
/// `TgmDecodeMaskOptions`.
unsafe fn apply_decode_mask_options(
    decode_opts: &mut DecodeOptions,
    opts: *const TgmDecodeMaskOptions,
) {
    if opts.is_null() {
        return;
    }
    let opts = unsafe { &*opts };
    decode_opts.restore_non_finite = opts.restore_non_finite;
}

/// Free a byte buffer returned by `tgm_encode`.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_bytes_free(buf: TgmBytes) {
    if !buf.data.is_null() {
        unsafe {
            drop(Vec::from_raw_parts(buf.data, buf.len, buf.len));
        }
    }
}

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

/// Decoded message: global metadata + decoded (descriptor, payload) pairs.
pub struct TgmMessage {
    global_metadata: GlobalMetadata,
    /// Each entry pairs a per-object descriptor with its decoded payload bytes.
    objects: Vec<(DataObjectDescriptor, Vec<u8>)>,
    /// Cached CStrings for dtype accessor returns (parallel to `objects`).
    dtype_strings: Vec<CString>,
    /// Cached CStrings for object type accessor returns.
    type_strings: Vec<CString>,
    /// Cached CStrings for byte order accessor returns.
    byte_order_strings: Vec<CString>,
    /// Cached CStrings for filter accessor returns.
    filter_strings: Vec<CString>,
    /// Cached CStrings for compression accessor returns.
    compression_strings: Vec<CString>,
    /// Cached CStrings for encoding accessor returns.
    encoding_strings: Vec<CString>,
    /// Cached CStrings for hash type accessor returns (None when no hash).
    hash_type_strings: Vec<Option<CString>>,
    /// Cached CStrings for hash value accessor returns (None when no hash).
    hash_value_strings: Vec<Option<CString>>,
}

/// Metadata-only handle (no decoded payloads).
pub struct TgmMetadata {
    global_metadata: GlobalMetadata,
    /// Cache for string accessors (key → null-terminated value).
    cache: std::cell::RefCell<BTreeMap<String, CString>>,
}

/// File handle.
pub struct TgmFile {
    file: TensogramFile,
    /// Cached path string for `tgm_file_path`.
    path_string: CString,
}

/// Scan result: array of (offset, length) pairs.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TgmScanEntry {
    pub offset: usize,
    pub length: usize,
}

/// Opaque handle for scan results.
pub struct TgmScanResult {
    entries: Vec<TgmScanEntry>,
}

// ---------------------------------------------------------------------------
// JSON deserialization helpers for the encode API
// ---------------------------------------------------------------------------

/// Intermediate struct used to parse the flat JSON provided to `tgm_encode`.
///
/// The caller passes a single JSON object that contains both global metadata
/// fields (`version`, and any extra namespaced keys such as `"mars"`) and a
/// `"descriptors"` array of per-object descriptor objects.
#[derive(serde::Deserialize)]
struct EncodeJson {
    version: u16,
    #[serde(default)]
    descriptors: Vec<DataObjectDescriptor>,
    /// Per-object metadata array (one entry per data object).
    #[serde(default)]
    base: Vec<BTreeMap<String, serde_json::Value>>,
    /// All remaining top-level keys become `GlobalMetadata::extra`.
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

/// Convert a `serde_json::Value` to a `ciborium::Value` for storage in
/// `GlobalMetadata::extra`.
fn json_to_cbor(v: serde_json::Value) -> ciborium::Value {
    match v {
        serde_json::Value::Null => ciborium::Value::Null,
        serde_json::Value::Bool(b) => ciborium::Value::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                ciborium::Value::Integer(i.into())
            } else if let Some(f) = n.as_f64() {
                ciborium::Value::Float(f)
            } else {
                ciborium::Value::Null
            }
        }
        serde_json::Value::String(s) => ciborium::Value::Text(s),
        serde_json::Value::Array(arr) => {
            ciborium::Value::Array(arr.into_iter().map(json_to_cbor).collect())
        }
        serde_json::Value::Object(map) => ciborium::Value::Map(
            map.into_iter()
                .map(|(k, v)| (ciborium::Value::Text(k), json_to_cbor(v)))
                .collect(),
        ),
    }
}

/// Parse the flat JSON blob into a `GlobalMetadata` and a list of
/// `DataObjectDescriptor`s.
///
/// The `"version"`, `"descriptors"`, and `"base"` keys are consumed; all
/// remaining keys are forwarded into `GlobalMetadata::extra` as CBOR values.
fn parse_encode_json(
    json_str: &str,
) -> Result<(GlobalMetadata, Vec<DataObjectDescriptor>), String> {
    let parsed: EncodeJson = serde_json::from_str(json_str)
        .map_err(|e| format!("failed to parse metadata JSON: {e}"))?;

    let cbor_base: Vec<BTreeMap<String, ciborium::Value>> = parsed
        .base
        .into_iter()
        .map(|entry| {
            entry
                .into_iter()
                .map(|(k, v)| (k, json_to_cbor(v)))
                .collect()
        })
        .collect();

    // Validate: no _reserved_ keys in base entries (library-managed namespace)
    for (i, entry) in cbor_base.iter().enumerate() {
        if entry.contains_key(RESERVED_KEY) {
            return Err(format!(
                "base[{i}] must not contain '{RESERVED_KEY}' key — the encoder populates it"
            ));
        }
    }

    let cbor_extra: BTreeMap<String, ciborium::Value> = parsed
        .extra
        .into_iter()
        .map(|(k, v)| (k, json_to_cbor(v)))
        .collect();

    let global_metadata = GlobalMetadata {
        version: parsed.version,
        base: cbor_base,
        extra: cbor_extra,
        ..Default::default()
    };

    Ok((global_metadata, parsed.descriptors))
}

// ---------------------------------------------------------------------------
// Message cache builder
// ---------------------------------------------------------------------------

/// Pre-built CString caches for all descriptor string fields.
struct MessageCaches {
    dtype_strings: Vec<CString>,
    type_strings: Vec<CString>,
    byte_order_strings: Vec<CString>,
    filter_strings: Vec<CString>,
    compression_strings: Vec<CString>,
    encoding_strings: Vec<CString>,
    hash_type_strings: Vec<Option<CString>>,
    hash_value_strings: Vec<Option<CString>>,
}

/// Build all CString caches from the object descriptors.
fn build_message_caches(objects: &[(DataObjectDescriptor, Vec<u8>)]) -> MessageCaches {
    let dtype_strings = objects
        .iter()
        .map(|(desc, _)| CString::new(desc.dtype.to_string()).unwrap_or_default())
        .collect();
    let type_strings = objects
        .iter()
        .map(|(desc, _)| CString::new(desc.obj_type.as_str()).unwrap_or_default())
        .collect();
    let byte_order_strings = objects
        .iter()
        .map(|(desc, _)| {
            let s = match desc.byte_order {
                tensogram::ByteOrder::Big => "big",
                tensogram::ByteOrder::Little => "little",
            };
            CString::new(s).unwrap_or_default()
        })
        .collect();
    let filter_strings = objects
        .iter()
        .map(|(desc, _)| CString::new(desc.filter.as_str()).unwrap_or_default())
        .collect();
    let compression_strings = objects
        .iter()
        .map(|(desc, _)| CString::new(desc.compression.as_str()).unwrap_or_default())
        .collect();
    let encoding_strings = objects
        .iter()
        .map(|(desc, _)| CString::new(desc.encoding.as_str()).unwrap_or_default())
        .collect();
    // v3: the per-object hash no longer lives in the CBOR
    // descriptor.  The inline slot in the frame footer is the
    // source of truth.  Until the FFI surfaces the inline slot
    // (phase 8), we return `None` for every descriptor so the
    // C API stops reporting stale values.
    let hash_type_strings: Vec<Option<CString>> = objects.iter().map(|_| None).collect();
    let hash_value_strings: Vec<Option<CString>> = objects.iter().map(|_| None).collect();

    MessageCaches {
        dtype_strings,
        type_strings,
        byte_order_strings,
        filter_strings,
        compression_strings,
        encoding_strings,
        hash_type_strings,
        hash_value_strings,
    }
}

// ---------------------------------------------------------------------------
// Shared encode argument parsing
// ---------------------------------------------------------------------------

/// Parsed and validated arguments shared by `tgm_encode` and `tgm_file_append`.
struct ParsedEncode<'a> {
    global_metadata: GlobalMetadata,
    descriptors: Vec<DataObjectDescriptor>,
    data_slices: Vec<&'a [u8]>,
    options: EncodeOptions,
}

/// Parse the hash algorithm from a nullable C string pointer.
///
/// Returns `Ok(None)` when the pointer is null, `Ok(Some(algo))` when valid,
/// or `Err((code, message))` on parse failure.
fn parse_hash_algo(hash_algo: *const c_char) -> Result<Option<HashAlgorithm>, (TgmError, String)> {
    if hash_algo.is_null() {
        return Ok(None);
    }
    let s = unsafe { CStr::from_ptr(hash_algo) }.to_str().map_err(|_| {
        (
            TgmError::InvalidArg,
            "invalid UTF-8 in hash_algo".to_string(),
        )
    })?;
    HashAlgorithm::parse(s)
        .map(Some)
        .map_err(|e| (TgmError::InvalidArg, e.to_string()))
}

/// Collect data slices from parallel C arrays with null-pointer validation.
///
/// # Safety
///
/// `data_ptrs` and `data_lens` must point to valid arrays of at least
/// `num_objects` elements. Each `data_ptrs[i]` must be valid for
/// `data_lens[i]` bytes (or may be null only when `data_lens[i] == 0`).
unsafe fn collect_data_slices<'a>(
    data_ptrs: *const *const u8,
    data_lens: *const usize,
    num_objects: usize,
) -> Result<Vec<&'a [u8]>, (TgmError, String)> {
    if num_objects == 0 {
        return Ok(vec![]);
    }
    if data_ptrs.is_null() || data_lens.is_null() {
        return Err((
            TgmError::InvalidArg,
            "null data_ptrs or data_lens".to_string(),
        ));
    }
    let ptrs = unsafe { slice::from_raw_parts(data_ptrs, num_objects) };
    let lens = unsafe { slice::from_raw_parts(data_lens, num_objects) };
    for (i, (&p, &l)) in ptrs.iter().zip(lens.iter()).enumerate() {
        if p.is_null() && l > 0 {
            return Err((
                TgmError::InvalidArg,
                format!("null data pointer at index {i}"),
            ));
        }
    }
    Ok(ptrs
        .iter()
        .zip(lens.iter())
        .map(|(&p, &l)| {
            if l == 0 {
                // Avoid calling slice::from_raw_parts with a potentially null
                // pointer when length is zero — that is UB even for zero-length
                // slices per the Rust reference.
                &[] as &[u8]
            } else {
                unsafe { slice::from_raw_parts(p, l) }
            }
        })
        .collect())
}

/// Parse and validate the common arguments for `tgm_encode` / `tgm_file_append`.
///
/// # Safety
///
/// All pointer arguments must satisfy the same contracts as the public FFI
/// functions that delegate to this helper.
unsafe fn parse_encode_args<'a>(
    json_str: &str,
    data_ptrs: *const *const u8,
    data_lens: *const usize,
    num_objects: usize,
    hash_algo: *const c_char,
    threads: u32,
) -> Result<ParsedEncode<'a>, (TgmError, String)> {
    let (global_metadata, descriptors) =
        parse_encode_json(json_str).map_err(|e| (TgmError::Metadata, e))?;

    if descriptors.len() != num_objects {
        return Err((
            TgmError::InvalidArg,
            format!(
                "descriptors array length {} does not match num_objects {}",
                descriptors.len(),
                num_objects
            ),
        ));
    }

    let data_slices = unsafe { collect_data_slices(data_ptrs, data_lens, num_objects) }?;
    let hash_algorithm = parse_hash_algo(hash_algo)?;
    let options = EncodeOptions {
        hash_algorithm,
        threads,
        ..Default::default()
    };

    Ok(ParsedEncode {
        global_metadata,
        descriptors,
        data_slices,
        options,
    })
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Encode a Tensogram message from JSON metadata and raw data slices.
///
/// `metadata_json`: null-terminated UTF-8 JSON string. Must contain:
///   - `"version"` (integer)
///   - `"descriptors"` (array of per-object descriptor objects)
///   - optional extra keys (e.g. `"mars"`) at the top level
///
/// `data_ptrs` / `data_lens`: arrays of length `num_objects`, raw bytes per object.
///
/// `hash_algo`: null-terminated string ("xxh3") or NULL for no hash.
///
/// On success returns `TgmError::Ok` and fills `out` with the encoded bytes.
/// The caller must free `out` with `tgm_bytes_free`.
///
/// 0.17+: encode rejects non-finite values (NaN / ±Inf) by default.
/// Use [`tgm_encode_with_options`] with a
/// [`TgmEncodeMaskOptions`] pointer (`allow_nan` / `allow_inf`) to
/// opt into NaN / Inf substitution with bitmask companion frames;
/// this entry point always uses the default reject policy.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_encode(
    metadata_json: *const c_char,
    data_ptrs: *const *const u8,
    data_lens: *const usize,
    num_objects: usize,
    hash_algo: *const c_char,
    threads: u32,
    out: *mut TgmBytes,
) -> TgmError {
    if metadata_json.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let json_str = match unsafe { CStr::from_ptr(metadata_json) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in metadata_json: {e}"));
            return TgmError::InvalidArg;
        }
    };

    let parsed = match unsafe {
        parse_encode_args(
            json_str,
            data_ptrs,
            data_lens,
            num_objects,
            hash_algo,
            threads,
        )
    } {
        Ok(p) => p,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };

    // Build (descriptor, data) pairs for the encode API
    let pairs: Vec<(&DataObjectDescriptor, &[u8])> = parsed
        .descriptors
        .iter()
        .zip(parsed.data_slices.iter())
        .map(|(d, s)| (d, *s))
        .collect();

    match encode(&parsed.global_metadata, &pairs, &parsed.options) {
        Ok(bytes) => {
            // Rebuild via boxed slice to guarantee capacity == len for tgm_bytes_free.
            let mut bytes = bytes.into_boxed_slice().into_vec();
            let result = TgmBytes {
                data: bytes.as_mut_ptr(),
                len: bytes.len(),
            };
            std::mem::forget(bytes); // ownership transferred to C
            unsafe {
                *out = result;
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

/// Encode with explicit NaN / Inf mask-companion options.
///
/// Like [`tgm_encode`] but takes a [`TgmEncodeMaskOptions`] pointer
/// (nullable — `NULL` behaves like [`tgm_encode`]'s default reject
/// policy).  All other arguments are identical.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn tgm_encode_with_options(
    metadata_json: *const c_char,
    data_ptrs: *const *const u8,
    data_lens: *const usize,
    num_objects: usize,
    hash_algo: *const c_char,
    threads: u32,
    mask_options: *const TgmEncodeMaskOptions,
    out: *mut TgmBytes,
) -> TgmError {
    if metadata_json.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let json_str = match unsafe { CStr::from_ptr(metadata_json) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in metadata_json: {e}"));
            return TgmError::InvalidArg;
        }
    };

    let mut parsed = match unsafe {
        parse_encode_args(
            json_str,
            data_ptrs,
            data_lens,
            num_objects,
            hash_algo,
            threads,
        )
    } {
        Ok(p) => p,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };
    if let Err(msg) = unsafe { apply_mask_options(&mut parsed.options, mask_options) } {
        set_last_error(&msg);
        return TgmError::InvalidArg;
    }

    let pairs: Vec<(&DataObjectDescriptor, &[u8])> = parsed
        .descriptors
        .iter()
        .zip(parsed.data_slices.iter())
        .map(|(d, s)| (d, *s))
        .collect();

    match encode(&parsed.global_metadata, &pairs, &parsed.options) {
        Ok(bytes) => {
            let mut bytes = bytes.into_boxed_slice().into_vec();
            let result = TgmBytes {
                data: bytes.as_mut_ptr(),
                len: bytes.len(),
            };
            std::mem::forget(bytes);
            unsafe {
                *out = result;
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

/// Decode with explicit NaN / Inf restoration options.
///
/// Like [`tgm_decode`] but takes a [`TgmDecodeMaskOptions`] pointer
/// (nullable — `NULL` behaves like [`tgm_decode`]'s default
/// `restore_non_finite = true`).
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn tgm_decode_with_options(
    buf: *const u8,
    buf_len: usize,
    verify_hash: i32,
    native_byte_order: i32,
    threads: u32,
    mask_options: *const TgmDecodeMaskOptions,
    out: *mut *mut TgmMessage,
) -> TgmError {
    if buf.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let data = unsafe { slice::from_raw_parts(buf, buf_len) };
    let mut options = DecodeOptions {
        verify_hash: verify_hash != 0,
        native_byte_order: native_byte_order != 0,
        threads,
        ..Default::default()
    };
    unsafe { apply_decode_mask_options(&mut options, mask_options) };

    match decode(data, &options) {
        Ok((global_metadata, objects)) => {
            let caches = build_message_caches(&objects);
            let msg = Box::new(TgmMessage {
                global_metadata,
                objects,
                dtype_strings: caches.dtype_strings,
                type_strings: caches.type_strings,
                byte_order_strings: caches.byte_order_strings,
                filter_strings: caches.filter_strings,
                compression_strings: caches.compression_strings,
                encoding_strings: caches.encoding_strings,
                hash_type_strings: caches.hash_type_strings,
                hash_value_strings: caches.hash_value_strings,
            });
            unsafe {
                *out = Box::into_raw(msg);
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

/// Streaming-encoder constructor with NaN / Inf mask-companion options.
///
/// Like [`tgm_streaming_encoder_create`] but takes a
/// [`TgmEncodeMaskOptions`] pointer.  `NULL` behaves like the default
/// reject policy.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn tgm_streaming_encoder_create_with_options(
    path: *const c_char,
    metadata_json: *const c_char,
    hash_algo: *const c_char,
    threads: u32,
    mask_options: *const TgmEncodeMaskOptions,
    out: *mut *mut TgmStreamingEncoder,
) -> TgmError {
    // Delegate to the existing tgm_streaming_encoder_create for the
    // validation + file-creation side-effects, then — if that
    // succeeded AND the caller passed a non-NULL mask_options — no
    // further adjustment is needed because the encoder has been
    // constructed with default options.  For an opt-in mask-aware
    // encoder we take a direct path through StreamingEncoder::new
    // with the full options set.
    //
    // Rationale for the direct path: the library-level EncodeOptions
    // is snapshotted at construction in StreamingEncoder, so we can't
    // retrofit mask options after the fact.  We therefore replicate
    // the existing validation + open logic here, pointer-for-pointer.
    if path.is_null() || metadata_json.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in path: {e}"));
            return TgmError::InvalidArg;
        }
    };
    let json_str = match unsafe { CStr::from_ptr(metadata_json) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in metadata_json: {e}"));
            return TgmError::InvalidArg;
        }
    };
    let global_metadata = match parse_streaming_metadata_json(json_str) {
        Ok(m) => m,
        Err(e) => {
            set_last_error(&e);
            return TgmError::Metadata;
        }
    };
    let hash_algorithm = if hash_algo.is_null() {
        None
    } else {
        let s = match unsafe { CStr::from_ptr(hash_algo) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                set_last_error("invalid UTF-8 in hash_algo");
                return TgmError::InvalidArg;
            }
        };
        match HashAlgorithm::parse(s) {
            Ok(a) => Some(a),
            Err(e) => {
                set_last_error(&e.to_string());
                return TgmError::InvalidArg;
            }
        }
    };
    let file = match std::fs::File::create(path_str) {
        Ok(f) => f,
        Err(e) => {
            set_last_error(&e.to_string());
            return TgmError::Io;
        }
    };
    let mut options = EncodeOptions {
        hash_algorithm,
        threads,
        ..Default::default()
    };
    if let Err(msg) = unsafe { apply_mask_options(&mut options, mask_options) } {
        set_last_error(&msg);
        return TgmError::InvalidArg;
    }
    let writer = std::io::BufWriter::new(file);
    match StreamingEncoder::new(writer, &global_metadata, &options) {
        Ok(enc) => {
            let handle = Box::new(TgmStreamingEncoder { inner: Some(enc) });
            unsafe {
                *out = Box::into_raw(handle);
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

/// Append a message to a file with explicit NaN / Inf mask-companion options.
///
/// Like [`tgm_file_append`] but takes a [`TgmEncodeMaskOptions`]
/// pointer.  `NULL` behaves like the default reject policy.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn tgm_file_append_with_options(
    file: *mut TgmFile,
    metadata_json: *const c_char,
    data_ptrs: *const *const u8,
    data_lens: *const usize,
    num_objects: usize,
    hash_algo: *const c_char,
    threads: u32,
    mask_options: *const TgmEncodeMaskOptions,
) -> TgmError {
    if file.is_null() || metadata_json.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let json_str = match unsafe { CStr::from_ptr(metadata_json) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in metadata_json: {e}"));
            return TgmError::InvalidArg;
        }
    };
    let mut parsed = match unsafe {
        parse_encode_args(
            json_str,
            data_ptrs,
            data_lens,
            num_objects,
            hash_algo,
            threads,
        )
    } {
        Ok(p) => p,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };
    if let Err(msg) = unsafe { apply_mask_options(&mut parsed.options, mask_options) } {
        set_last_error(&msg);
        return TgmError::InvalidArg;
    }
    let pairs: Vec<(&DataObjectDescriptor, &[u8])> = parsed
        .descriptors
        .iter()
        .zip(parsed.data_slices.iter())
        .map(|(d, s)| (d, *s))
        .collect();
    let f = unsafe { &mut (*file).file };
    match f.append(&parsed.global_metadata, &pairs, &parsed.options) {
        Ok(()) => TgmError::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

/// Encode a Tensogram message from JSON metadata and pre-encoded payload bytes.
///
/// Like `tgm_encode`, but each `data_ptrs[i]` slice must already be encoded
/// according to the matching descriptor's `encoding` / `filter` / `compression`
/// pipeline. The library does not run the encoding pipeline again — it writes
/// the caller-provided bytes directly into the wire-format payload after
/// validating that the descriptor's pipeline configuration is well-formed.
///
/// `metadata_json`: same flat JSON schema as `tgm_encode` (`version`,
///   `descriptors`, optional `base`, plus arbitrary extra top-level keys).
///
/// `data_ptrs` / `data_lens`: arrays of length `num_objects` pointing at
///   already-encoded payload bytes (one entry per descriptor).
///
/// `hash_algo`: null-terminated string ("xxh3") or NULL for no hash. The
///   library always recomputes the hash over the caller's bytes; any
///   `hash` field embedded in the descriptor JSON is ignored and overwritten.
///
/// Notes for compression-aware decoding:
/// - For `szip` compression, callers SHOULD include `szip_block_offsets`
///   (a list of bit offsets into the compressed payload) inside the
///   matching descriptor's params so that `tgm_decode_range` can locate
///   szip block boundaries without rescanning the compressed stream.
/// - Other pipeline params (e.g. `simple_packing` reference value, scale
///   factors) must also be present in the descriptor — they are not
///   inferred from the bytes.
///
/// On success returns `TgmError::Ok` and fills `out` with the encoded message.
/// The caller must free `out` with `tgm_bytes_free`.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_encode_pre_encoded(
    metadata_json: *const c_char,
    data_ptrs: *const *const u8,
    data_lens: *const usize,
    num_objects: usize,
    hash_algo: *const c_char,
    threads: u32,
    out: *mut TgmBytes,
) -> TgmError {
    if metadata_json.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let json_str = match unsafe { CStr::from_ptr(metadata_json) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in metadata_json: {e}"));
            return TgmError::InvalidArg;
        }
    };

    let parsed = match unsafe {
        parse_encode_args(
            json_str,
            data_ptrs,
            data_lens,
            num_objects,
            hash_algo,
            threads,
        )
    } {
        Ok(p) => p,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };

    // Build (descriptor, pre-encoded data) pairs for the pre-encoded API.
    let pairs: Vec<(&DataObjectDescriptor, &[u8])> = parsed
        .descriptors
        .iter()
        .zip(parsed.data_slices.iter())
        .map(|(d, s)| (d, *s))
        .collect();

    match encode_pre_encoded(&parsed.global_metadata, &pairs, &parsed.options) {
        Ok(bytes) => {
            // Rebuild via boxed slice to guarantee capacity == len for tgm_bytes_free.
            let mut bytes = bytes.into_boxed_slice().into_vec();
            let result = TgmBytes {
                data: bytes.as_mut_ptr(),
                len: bytes.len(),
            };
            std::mem::forget(bytes); // ownership transferred to C
            unsafe {
                *out = result;
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Decode a complete message (global metadata + all object payloads).
///
/// `buf` / `buf_len`: the wire-format message bytes.
/// `verify_hash`: if non-zero, verify payload hashes during decode.
///
/// On success, fills `out` with a `TgmMessage` handle.
/// Free with `tgm_message_free`.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_decode(
    buf: *const u8,
    buf_len: usize,
    verify_hash: i32,
    native_byte_order: i32,
    threads: u32,
    out: *mut *mut TgmMessage,
) -> TgmError {
    if buf.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let data = unsafe { slice::from_raw_parts(buf, buf_len) };
    let options = DecodeOptions {
        verify_hash: verify_hash != 0,
        native_byte_order: native_byte_order != 0,
        threads,
        ..Default::default()
    };

    match decode(data, &options) {
        Ok((global_metadata, objects)) => {
            let caches = build_message_caches(&objects);
            let msg = Box::new(TgmMessage {
                global_metadata,
                objects,
                dtype_strings: caches.dtype_strings,
                type_strings: caches.type_strings,
                byte_order_strings: caches.byte_order_strings,
                filter_strings: caches.filter_strings,
                compression_strings: caches.compression_strings,
                encoding_strings: caches.encoding_strings,
                hash_type_strings: caches.hash_type_strings,
                hash_value_strings: caches.hash_value_strings,
            });
            unsafe {
                *out = Box::into_raw(msg);
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

/// Decode only the global metadata (no payload bytes are read).
#[unsafe(no_mangle)]
pub extern "C" fn tgm_decode_metadata(
    buf: *const u8,
    buf_len: usize,
    out: *mut *mut TgmMetadata,
) -> TgmError {
    if buf.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let data = unsafe { slice::from_raw_parts(buf, buf_len) };

    match decode_metadata(data) {
        Ok(global_metadata) => {
            let m = Box::new(TgmMetadata {
                global_metadata,
                cache: std::cell::RefCell::new(BTreeMap::new()),
            });
            unsafe {
                *out = Box::into_raw(m);
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

/// Decode a single object by index.
///
/// On success, fills `out` with a `TgmMessage` handle containing exactly
/// one object (at index 0). The global metadata covers the whole message.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_decode_object(
    buf: *const u8,
    buf_len: usize,
    index: usize,
    verify_hash: i32,
    native_byte_order: i32,
    threads: u32,
    out: *mut *mut TgmMessage,
) -> TgmError {
    if buf.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let data = unsafe { slice::from_raw_parts(buf, buf_len) };
    let options = DecodeOptions {
        verify_hash: verify_hash != 0,
        native_byte_order: native_byte_order != 0,
        threads,
        ..Default::default()
    };

    match decode_object(data, index, &options) {
        Ok((global_metadata, descriptor, obj_bytes)) => {
            let objects = vec![(descriptor, obj_bytes)];
            let caches = build_message_caches(&objects);
            let msg = Box::new(TgmMessage {
                global_metadata,
                objects,
                dtype_strings: caches.dtype_strings,
                type_strings: caches.type_strings,
                byte_order_strings: caches.byte_order_strings,
                filter_strings: caches.filter_strings,
                compression_strings: caches.compression_strings,
                encoding_strings: caches.encoding_strings,
                hash_type_strings: caches.hash_type_strings,
                hash_value_strings: caches.hash_value_strings,
            });
            unsafe {
                *out = Box::into_raw(msg);
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

/// Decode partial ranges from a data object.
///
/// `ranges_offsets` / `ranges_counts`: parallel arrays of (element_offset, element_count).
/// `num_ranges`: length of both arrays.
/// `join`: when non-zero, concatenate all ranges into a single buffer in `out[0]`
///         and set `*out_count = 1`.  When zero (split mode), write one `TgmBytes`
///         per range into `out[0..num_ranges]` and set `*out_count = num_ranges`.
///         The caller must pre-allocate `out` with at least `num_ranges` entries
///         when `join == 0`, or 1 entry when `join != 0`.
/// `out_count`: filled with the number of buffers written to `out`.
///
/// Free each returned buffer with `tgm_bytes_free`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn tgm_decode_range(
    buf: *const u8,
    buf_len: usize,
    object_index: usize,
    ranges_offsets: *const u64,
    ranges_counts: *const u64,
    num_ranges: usize,
    verify_hash: i32,
    native_byte_order: i32,
    threads: u32,
    join: i32,
    out: *mut TgmBytes,
    out_count: *mut usize,
) -> TgmError {
    if buf.is_null() || out.is_null() || out_count.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    if num_ranges > 0 && (ranges_offsets.is_null() || ranges_counts.is_null()) {
        set_last_error("null ranges_offsets or ranges_counts");
        return TgmError::InvalidArg;
    }

    let data = unsafe { slice::from_raw_parts(buf, buf_len) };
    let options = DecodeOptions {
        verify_hash: verify_hash != 0,
        native_byte_order: native_byte_order != 0,
        threads,
        ..Default::default()
    };

    let ranges: Vec<(u64, u64)> = if num_ranges == 0 {
        vec![]
    } else {
        unsafe {
            let offsets = slice::from_raw_parts(ranges_offsets, num_ranges);
            let counts = slice::from_raw_parts(ranges_counts, num_ranges);
            offsets
                .iter()
                .zip(counts.iter())
                .map(|(&o, &c)| (o, c))
                .collect()
        }
    };

    match decode_range(data, object_index, &ranges, &options) {
        Ok((_, parts)) => {
            if join != 0 {
                // Concatenate all parts into a single buffer.
                let joined: Vec<u8> = parts.into_iter().flatten().collect();
                let mut joined = joined.into_boxed_slice().into_vec();
                let result = TgmBytes {
                    data: joined.as_mut_ptr(),
                    len: joined.len(),
                };
                std::mem::forget(joined);
                unsafe {
                    *out = result;
                    *out_count = 1;
                }
            } else {
                // Write one TgmBytes per range.
                let n = parts.len();
                for (i, part) in parts.into_iter().enumerate() {
                    let mut part = part.into_boxed_slice().into_vec();
                    let result = TgmBytes {
                        data: part.as_mut_ptr(),
                        len: part.len(),
                    };
                    std::mem::forget(part);
                    unsafe {
                        *out.add(i) = result;
                    }
                }
                unsafe {
                    *out_count = n;
                }
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

/// Scan a buffer for message boundaries.
///
/// Returns a `TgmScanResult` handle. Access entries with `tgm_scan_count`
/// and `tgm_scan_entry`. Free with `tgm_scan_free`.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_scan(
    buf: *const u8,
    buf_len: usize,
    out: *mut *mut TgmScanResult,
) -> TgmError {
    if buf.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let data = unsafe { slice::from_raw_parts(buf, buf_len) };
    let offsets = scan(data);
    let entries: Vec<TgmScanEntry> = offsets
        .into_iter()
        .map(|(offset, length)| TgmScanEntry { offset, length })
        .collect();
    let result = Box::new(TgmScanResult { entries });
    unsafe {
        *out = Box::into_raw(result);
    }
    TgmError::Ok
}

/// # Safety: caller must pass valid, non-null pointer from tgm_scan.
unsafe fn as_scan(result: *const TgmScanResult) -> Option<&'static TgmScanResult> {
    unsafe {
        if result.is_null() {
            None
        } else {
            Some(&*result)
        }
    }
}

/// # Safety: caller must pass valid, non-null pointer from tgm_decode*.
unsafe fn as_msg(msg: *const TgmMessage) -> Option<&'static TgmMessage> {
    unsafe { if msg.is_null() { None } else { Some(&*msg) } }
}

/// Returns the number of messages found by `tgm_scan`.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_scan_count(result: *const TgmScanResult) -> usize {
    unsafe { as_scan(result).map(|r| r.entries.len()).unwrap_or(0) }
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_scan_entry(result: *const TgmScanResult, index: usize) -> TgmScanEntry {
    let fallback = TgmScanEntry {
        offset: usize::MAX,
        length: 0,
    };
    unsafe {
        match as_scan(result) {
            Some(r) => match r.entries.get(index) {
                Some(entry) => *entry,
                None => {
                    set_last_error(&format!(
                        "scan entry index {} out of range (count={})",
                        index,
                        r.entries.len()
                    ));
                    fallback
                }
            },
            None => {
                set_last_error("null scan result handle");
                fallback
            }
        }
    }
}

/// Free a scan result handle.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_scan_free(result: *mut TgmScanResult) {
    if !result.is_null() {
        unsafe {
            drop(Box::from_raw(result));
        }
    }
}

// ---------------------------------------------------------------------------
// Message accessors
// ---------------------------------------------------------------------------

/// Returns the wire format version from a decoded message.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_message_version(msg: *const TgmMessage) -> u64 {
    unsafe {
        as_msg(msg)
            .map(|m| m.global_metadata.version as u64)
            .unwrap_or(0)
    }
}

/// Returns the number of decoded objects in this message handle.
/// For `tgm_decode` this equals the total object count; for
/// `tgm_decode_object` this is always 1.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_message_num_objects(msg: *const TgmMessage) -> usize {
    unsafe { as_msg(msg).map(|m| m.objects.len()).unwrap_or(0) }
}

/// Returns the number of decoded payload buffers.
/// Equivalent to `tgm_message_num_objects` — kept for ABI compatibility.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_message_num_decoded(msg: *const TgmMessage) -> usize {
    unsafe { as_msg(msg).map(|m| m.objects.len()).unwrap_or(0) }
}

/// Returns the number of dimensions for object at index.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_object_ndim(msg: *const TgmMessage, index: usize) -> u64 {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.objects.get(index))
            .map(|(desc, _)| desc.ndim)
            .unwrap_or(0)
    }
}

/// Returns a pointer to the shape array. Length is `tgm_object_ndim()`.
/// The pointer is valid until the message is freed.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_object_shape(msg: *const TgmMessage, index: usize) -> *const u64 {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.objects.get(index))
            .map(|(desc, _)| desc.shape.as_ptr())
            .unwrap_or(ptr::null())
    }
}

/// Returns a pointer to the strides array. Length is `tgm_object_ndim()`.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_object_strides(msg: *const TgmMessage, index: usize) -> *const u64 {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.objects.get(index))
            .map(|(desc, _)| desc.strides.as_ptr())
            .unwrap_or(ptr::null())
    }
}

/// Returns the dtype as a null-terminated string (e.g. "float32").
/// The pointer is valid until the message is freed.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_object_dtype(msg: *const TgmMessage, index: usize) -> *const c_char {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.dtype_strings.get(index))
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    }
}

/// Returns a pointer to the decoded payload bytes for a decoded object.
/// `decoded_index` is the index into the decoded objects array (0 for the
/// first decoded object, regardless of the original object index).
/// `out_len` receives the byte length.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_object_data(
    msg: *const TgmMessage,
    decoded_index: usize,
    out_len: *mut usize,
) -> *const u8 {
    unsafe {
        match as_msg(msg).and_then(|m| m.objects.get(decoded_index)) {
            Some((_, data)) => {
                if !out_len.is_null() {
                    *out_len = data.len();
                }
                data.as_ptr()
            }
            None => {
                if !out_len.is_null() {
                    *out_len = 0;
                }
                ptr::null()
            }
        }
    }
}

/// Returns the encoding string for a data object descriptor (e.g. "none", "simple_packing").
/// The pointer is valid until the message is freed.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_payload_encoding(msg: *const TgmMessage, index: usize) -> *const c_char {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.encoding_strings.get(index))
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    }
}

/// Returns 1 if the object descriptor has a hash, 0 otherwise.
///
/// **v3 deprecation.** The per-object hash has moved from the CBOR
/// descriptor to the frame footer's inline slot (see
/// `plans/WIRE_FORMAT.md` §2.4).  This FFI entry point always
/// returns `0` in v3 pending the phase-8 binding update that will
/// surface the inline slot (and the message-level
/// `HASHES_PRESENT` flag) through a new API.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_payload_has_hash(msg: *const TgmMessage, _index: usize) -> i32 {
    // Intentionally ignores `msg` / `_index`: the stale-surface
    // contract is "v3 descriptors never carry a hash".
    let _ = msg;
    0
}

/// Extract a metadata handle from a decoded message.
/// The metadata handle is independent — free it separately with `tgm_metadata_free`.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_message_metadata(
    msg: *const TgmMessage,
    out: *mut *mut TgmMetadata,
) -> TgmError {
    if msg.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let m = unsafe { &*msg };
    let meta = Box::new(TgmMetadata {
        global_metadata: m.global_metadata.clone(),
        cache: std::cell::RefCell::new(BTreeMap::new()),
    });
    unsafe {
        *out = Box::into_raw(meta);
    }
    TgmError::Ok
}

/// Returns the object type string (e.g. "ndarray"). Valid until message freed.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_object_type(msg: *const TgmMessage, index: usize) -> *const c_char {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.type_strings.get(index))
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    }
}

/// Returns the byte order string ("big" or "little"). Valid until message freed.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_object_byte_order(msg: *const TgmMessage, index: usize) -> *const c_char {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.byte_order_strings.get(index))
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    }
}

/// Returns the filter string (e.g. "none", "shuffle"). Valid until message freed.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_object_filter(msg: *const TgmMessage, index: usize) -> *const c_char {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.filter_strings.get(index))
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    }
}

/// Returns the compression string (e.g. "none", "zstd"). Valid until message freed.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_object_compression(msg: *const TgmMessage, index: usize) -> *const c_char {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.compression_strings.get(index))
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    }
}

/// Returns the hash type string ("xxh3") or NULL if no hash. Valid until message freed.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_object_hash_type(msg: *const TgmMessage, index: usize) -> *const c_char {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.hash_type_strings.get(index))
            .and_then(|opt| opt.as_ref())
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    }
}

/// Returns the hash value hex string or NULL if no hash. Valid until message freed.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_object_hash_value(msg: *const TgmMessage, index: usize) -> *const c_char {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.hash_value_strings.get(index))
            .and_then(|opt| opt.as_ref())
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    }
}

/// Free a decoded message handle.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_message_free(msg: *mut TgmMessage) {
    if !msg.is_null() {
        unsafe {
            drop(Box::from_raw(msg));
        }
    }
}

// ---------------------------------------------------------------------------
// Metadata accessors
// ---------------------------------------------------------------------------

/// Returns the wire format version from metadata.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_metadata_version(meta: *const TgmMetadata) -> u64 {
    if meta.is_null() {
        return 0;
    }
    unsafe { (*meta).global_metadata.version as u64 }
}

/// Returns the number of objects described in the global metadata.
///
/// Returns the length of the `base` array, which has one entry per data object.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_metadata_num_objects(meta: *const TgmMetadata) -> usize {
    if meta.is_null() {
        return 0;
    }
    unsafe { (*meta).global_metadata.base.len() }
}

/// Look up a string value by dot-notation key (e.g. "mars.class").
/// Returns NULL if the key is not found or is not a string.
/// The pointer is valid until the metadata handle is freed.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_metadata_get_string(
    meta: *const TgmMetadata,
    key: *const c_char,
) -> *const c_char {
    if meta.is_null() || key.is_null() {
        return ptr::null();
    }

    let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null(),
    };

    let m = unsafe { &(*meta) };
    let value = lookup_string_key(&m.global_metadata, key_str);

    match value {
        Some(s) => {
            let mut cache = m.cache.borrow_mut();
            let entry = cache
                .entry(key_str.to_string())
                .or_insert_with(|| CString::new(s.clone()).unwrap_or_default());
            entry.as_ptr()
        }
        None => ptr::null(),
    }
}

/// Look up an integer value by dot-notation key.
/// Returns `default_val` if the key is not found or is not an integer.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_metadata_get_int(
    meta: *const TgmMetadata,
    key: *const c_char,
    default_val: i64,
) -> i64 {
    if meta.is_null() || key.is_null() {
        return default_val;
    }

    let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
        Ok(s) => s,
        Err(_) => return default_val,
    };

    let m = unsafe { &(*meta) };
    lookup_int_key(&m.global_metadata, key_str).unwrap_or(default_val)
}

/// Look up a float value by dot-notation key.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_metadata_get_float(
    meta: *const TgmMetadata,
    key: *const c_char,
    default_val: f64,
) -> f64 {
    if meta.is_null() || key.is_null() {
        return default_val;
    }

    let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
        Ok(s) => s,
        Err(_) => return default_val,
    };

    let m = unsafe { &(*meta) };
    lookup_float_key(&m.global_metadata, key_str).unwrap_or(default_val)
}

/// Free a metadata handle.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_metadata_free(meta: *mut TgmMetadata) {
    if !meta.is_null() {
        unsafe {
            drop(Box::from_raw(meta));
        }
    }
}

// ---------------------------------------------------------------------------
// File API
// ---------------------------------------------------------------------------

/// Open an existing Tensogram file for reading.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_file_open(path: *const c_char, out: *mut *mut TgmFile) -> TgmError {
    if path.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in path: {e}"));
            return TgmError::InvalidArg;
        }
    };

    match TensogramFile::open(path_str) {
        Ok(file) => {
            let path_string = CString::new(path_str).unwrap_or_default();
            let handle = Box::new(TgmFile { file, path_string });
            unsafe {
                *out = Box::into_raw(handle);
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

/// Create a new Tensogram file for writing.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_file_create(path: *const c_char, out: *mut *mut TgmFile) -> TgmError {
    if path.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in path: {e}"));
            return TgmError::InvalidArg;
        }
    };

    match TensogramFile::create(path_str) {
        Ok(file) => {
            let path_string = CString::new(path_str).unwrap_or_default();
            let handle = Box::new(TgmFile { file, path_string });
            unsafe {
                *out = Box::into_raw(handle);
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

/// Count messages in the file (may trigger lazy scan).
#[unsafe(no_mangle)]
pub extern "C" fn tgm_file_message_count(file: *mut TgmFile, out_count: *mut usize) -> TgmError {
    if file.is_null() || out_count.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let f = unsafe { &(*file).file };
    match f.message_count() {
        Ok(count) => {
            unsafe {
                *out_count = count;
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

/// Decode message at `index` from the file.
/// On success fills `out` with a `TgmMessage` handle.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_file_decode_message(
    file: *mut TgmFile,
    index: usize,
    verify_hash: i32,
    native_byte_order: i32,
    threads: u32,
    out: *mut *mut TgmMessage,
) -> TgmError {
    if file.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let f = unsafe { &(*file).file };
    let options = DecodeOptions {
        verify_hash: verify_hash != 0,
        native_byte_order: native_byte_order != 0,
        threads,
        ..Default::default()
    };

    match f.decode_message(index, &options) {
        Ok((global_metadata, objects)) => {
            let caches = build_message_caches(&objects);
            let msg = Box::new(TgmMessage {
                global_metadata,
                objects,
                dtype_strings: caches.dtype_strings,
                type_strings: caches.type_strings,
                byte_order_strings: caches.byte_order_strings,
                filter_strings: caches.filter_strings,
                compression_strings: caches.compression_strings,
                encoding_strings: caches.encoding_strings,
                hash_type_strings: caches.hash_type_strings,
                hash_value_strings: caches.hash_value_strings,
            });
            unsafe {
                *out = Box::into_raw(msg);
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

/// Read raw message bytes at `index`.
/// On success fills `out` with a `TgmBytes` buffer.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_file_read_message(
    file: *mut TgmFile,
    index: usize,
    out: *mut TgmBytes,
) -> TgmError {
    if file.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let f = unsafe { &(*file).file };

    match f.read_message(index) {
        Ok(bytes) => {
            // Rebuild via boxed slice to guarantee capacity == len for tgm_bytes_free.
            let mut bytes = bytes.into_boxed_slice().into_vec();
            let result = TgmBytes {
                data: bytes.as_mut_ptr(),
                len: bytes.len(),
            };
            std::mem::forget(bytes);
            unsafe {
                *out = result;
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

/// Append raw message bytes to the file.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_file_append_raw(
    file: *mut TgmFile,
    buf: *const u8,
    buf_len: usize,
) -> TgmError {
    if file.is_null() || buf.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let data = unsafe { slice::from_raw_parts(buf, buf_len) };
    let f = unsafe { &mut (*file).file };

    // Write raw bytes using std::fs
    use std::io::Write;
    let path = match f.path() {
        Some(p) => p.to_path_buf(),
        None => {
            set_last_error("append_raw not supported on remote files");
            return TgmError::Remote;
        }
    };
    let result = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut fh| fh.write_all(data));

    match result {
        Ok(()) => {
            f.invalidate_offsets();
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            TgmError::Io
        }
    }
}

/// Returns the file path as a null-terminated string.
/// The pointer is valid until the file handle is closed.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_file_path(file: *const TgmFile) -> *const c_char {
    if file.is_null() {
        return ptr::null();
    }
    unsafe { (*file).path_string.as_ptr() }
}

/// Encode and append a message to the file.
/// Same JSON schema as `tgm_encode` for `metadata_json`.
///
/// Non-finite-value rejection is on by default in 0.17+; see `tgm_encode`.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_file_append(
    file: *mut TgmFile,
    metadata_json: *const c_char,
    data_ptrs: *const *const u8,
    data_lens: *const usize,
    num_objects: usize,
    hash_algo: *const c_char,
    threads: u32,
) -> TgmError {
    if file.is_null() || metadata_json.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let json_str = match unsafe { CStr::from_ptr(metadata_json) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in metadata_json: {e}"));
            return TgmError::InvalidArg;
        }
    };

    let parsed = match unsafe {
        parse_encode_args(
            json_str,
            data_ptrs,
            data_lens,
            num_objects,
            hash_algo,
            threads,
        )
    } {
        Ok(p) => p,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };

    let pairs: Vec<(&DataObjectDescriptor, &[u8])> = parsed
        .descriptors
        .iter()
        .zip(parsed.data_slices.iter())
        .map(|(d, s)| (d, *s))
        .collect();

    let f = unsafe { &mut (*file).file };
    match f.append(&parsed.global_metadata, &pairs, &parsed.options) {
        Ok(()) => TgmError::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

/// Close a file handle and release resources.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_file_close(file: *mut TgmFile) {
    if !file.is_null() {
        unsafe {
            drop(Box::from_raw(file));
        }
    }
}

// ---------------------------------------------------------------------------
// Metadata key lookup helpers
// ---------------------------------------------------------------------------

/// Look up a CBOR value by dot-notation key with arbitrary nesting depth.
///
/// Supports `"key"`, `"ns.field"`, `"grib.geography.Ni"`, etc.
/// Search order: `base[i]` (skip `_reserved_`, first match) → `extra`.
fn lookup_cbor_value<'a>(
    global_metadata: &'a GlobalMetadata,
    key: &str,
) -> Option<&'a ciborium::Value> {
    if key.is_empty() {
        return None;
    }
    let parts: Vec<&str> = key.split('.').collect();

    if parts.is_empty() || parts[0].is_empty() {
        return None;
    }
    if parts[0] == "version" {
        return None; // use tgm_metadata_version instead
    }

    // Explicit _extra_ or extra prefix targets the extra map directly
    if parts[0] == "_extra_" || parts[0] == "extra" {
        if parts.len() > 1 {
            return resolve_in_btree(&global_metadata.extra, &parts[1..]);
        }
        return None;
    }

    // Search base entries (skip _reserved_ key within each entry)
    for entry in &global_metadata.base {
        if let Some(val) = resolve_in_btree_skip_reserved(entry, &parts) {
            return Some(val);
        }
    }
    // Fall back to extra
    resolve_in_btree(&global_metadata.extra, &parts)
}

/// Walk a dot-path in a BTreeMap, skipping `_reserved_` keys at the first level.
fn resolve_in_btree_skip_reserved<'a>(
    map: &'a BTreeMap<String, ciborium::Value>,
    parts: &[&str],
) -> Option<&'a ciborium::Value> {
    let (first, rest) = parts.split_first()?;
    if *first == RESERVED_KEY {
        return None;
    }
    let value = map.get(*first)?;
    resolve_cbor_path(value, rest)
}

/// Walk a dot-path in a BTreeMap (no _reserved_ filtering).
fn resolve_in_btree<'a>(
    map: &'a BTreeMap<String, ciborium::Value>,
    parts: &[&str],
) -> Option<&'a ciborium::Value> {
    let (first, rest) = parts.split_first()?;
    let value = map.get(*first)?;
    resolve_cbor_path(value, rest)
}

/// Recursively walk remaining path segments into a CBOR value.
///
/// When no segments remain, returns the current value.
/// When segments remain, the current value must be a `Map` to navigate further.
fn resolve_cbor_path<'a>(
    value: &'a ciborium::Value,
    remaining: &[&str],
) -> Option<&'a ciborium::Value> {
    if remaining.is_empty() {
        return Some(value);
    }
    if let ciborium::Value::Map(entries) = value {
        for (k, v) in entries {
            if matches!(k, ciborium::Value::Text(s) if s == remaining[0]) {
                return resolve_cbor_path(v, &remaining[1..]);
            }
        }
    }
    None
}

fn lookup_string_key(global_metadata: &GlobalMetadata, key: &str) -> Option<String> {
    if key.is_empty() {
        return None;
    }
    if key == "version" {
        return Some(global_metadata.version.to_string());
    }

    lookup_cbor_value(global_metadata, key).and_then(|v| match v {
        ciborium::Value::Text(s) => Some(s.clone()),
        ciborium::Value::Integer(i) => {
            let n: i128 = (*i).into();
            Some(n.to_string())
        }
        ciborium::Value::Float(f) => Some(f.to_string()),
        ciborium::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    })
}

fn lookup_int_key(global_metadata: &GlobalMetadata, key: &str) -> Option<i64> {
    if key == "version" {
        return Some(global_metadata.version as i64);
    }

    lookup_cbor_value(global_metadata, key).and_then(|v| match v {
        ciborium::Value::Integer(i) => {
            let n: i128 = (*i).into();
            i64::try_from(n).ok()
        }
        _ => None,
    })
}

fn lookup_float_key(global_metadata: &GlobalMetadata, key: &str) -> Option<f64> {
    lookup_cbor_value(global_metadata, key).and_then(|v| match v {
        ciborium::Value::Float(f) => Some(*f),
        ciborium::Value::Integer(i) => {
            let n: i128 = (*i).into();
            // i128 → f64 may lose precision for very large integers, but this
            // is the expected behavior for a float accessor on an integer value.
            Some(n as f64)
        }
        _ => None,
    })
}

// ---------------------------------------------------------------------------
// simple_packing direct access
// ---------------------------------------------------------------------------

/// Compute simple_packing parameters for a set of f64 values.
///
/// Returns TgmError::Ok on success, filling the out-params.
/// Returns Encoding error if data contains NaN.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_simple_packing_compute_params(
    values: *const f64,
    num_values: usize,
    bits_per_value: u32,
    decimal_scale_factor: i32,
    out_reference_value: *mut f64,
    out_binary_scale_factor: *mut i32,
) -> TgmError {
    if values.is_null() || out_reference_value.is_null() || out_binary_scale_factor.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let vals = unsafe { slice::from_raw_parts(values, num_values) };

    match tensogram_encodings::simple_packing::compute_params(
        vals,
        bits_per_value,
        decimal_scale_factor,
    ) {
        Ok(params) => {
            unsafe {
                *out_reference_value = params.reference_value;
                *out_binary_scale_factor = params.binary_scale_factor;
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            TgmError::Encoding
        }
    }
}

// ---------------------------------------------------------------------------
// Iterator API
// ---------------------------------------------------------------------------

/// Opaque handle for iterating over messages in a byte buffer.
///
/// The caller's buffer must remain valid for the lifetime of this iterator.
pub struct TgmBufferIter {
    offsets: Vec<(usize, usize)>,
    buf_ptr: *const u8,
    pos: usize,
}

/// Create a buffer message iterator.
///
/// Scans `buf` once and stores message boundaries. The buffer must remain
/// valid and unmodified until `tgm_buffer_iter_free` is called.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_buffer_iter_create(
    buf: *const u8,
    buf_len: usize,
    out: *mut *mut TgmBufferIter,
) -> TgmError {
    if buf.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let data = unsafe { slice::from_raw_parts(buf, buf_len) };
    let offsets = scan(data);
    let iter = Box::new(TgmBufferIter {
        offsets,
        buf_ptr: buf,
        pos: 0,
    });
    unsafe {
        *out = Box::into_raw(iter);
    }
    TgmError::Ok
}

/// Return the total number of messages in the buffer iterator.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_buffer_iter_count(iter: *const TgmBufferIter) -> usize {
    if iter.is_null() {
        return 0;
    }
    unsafe { (*iter).offsets.len() }
}

/// Advance the buffer iterator. On success, sets `out_buf` and `out_len` to
/// the next message slice (borrowed from the original buffer).
///
/// Returns `TgmError::Ok` if a message is available, `TgmError::EndOfIter`
/// when iteration is exhausted.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_buffer_iter_next(
    iter: *mut TgmBufferIter,
    out_buf: *mut *const u8,
    out_len: *mut usize,
) -> TgmError {
    if iter.is_null() || out_buf.is_null() || out_len.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let it = unsafe { &mut *iter };
    if it.pos >= it.offsets.len() {
        return TgmError::EndOfIter;
    }
    let (offset, length) = it.offsets[it.pos];
    it.pos += 1;
    unsafe {
        *out_buf = it.buf_ptr.add(offset);
        *out_len = length;
    }
    TgmError::Ok
}

/// Free a buffer iterator handle.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_buffer_iter_free(iter: *mut TgmBufferIter) {
    if !iter.is_null() {
        unsafe {
            drop(Box::from_raw(iter));
        }
    }
}

/// Opaque handle for iterating over messages in a file.
pub struct TgmFileIter {
    inner: tensogram::FileMessageIter,
}

/// Create a file message iterator from an open TgmFile.
///
/// Scans the file to locate message boundaries. The file handle remains
/// usable after this call.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_file_iter_create(file: *mut TgmFile, out: *mut *mut TgmFileIter) -> TgmError {
    if file.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let f = unsafe { &(*file).file };
    match f.iter() {
        Ok(inner) => {
            let iter = Box::new(TgmFileIter { inner });
            unsafe {
                *out = Box::into_raw(iter);
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

/// Advance the file iterator. On success, fills `out` with a `TgmBytes`
/// buffer containing the raw message bytes (caller owns, free with
/// `tgm_bytes_free`).
///
/// Returns `TgmError::Ok` when a message is available, `TgmError::EndOfIter`
/// when iteration is exhausted.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_file_iter_next(iter: *mut TgmFileIter, out: *mut TgmBytes) -> TgmError {
    if iter.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let it = unsafe { &mut (*iter).inner };
    match it.next() {
        None => TgmError::EndOfIter,
        Some(Err(e)) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
        Some(Ok(bytes)) => {
            // Rebuild via boxed slice to guarantee capacity == len for tgm_bytes_free.
            let mut bytes = bytes.into_boxed_slice().into_vec();
            let result = TgmBytes {
                data: bytes.as_mut_ptr(),
                len: bytes.len(),
            };
            std::mem::forget(bytes);
            unsafe {
                *out = result;
            }
            TgmError::Ok
        }
    }
}

/// Free a file iterator handle.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_file_iter_free(iter: *mut TgmFileIter) {
    if !iter.is_null() {
        unsafe {
            drop(Box::from_raw(iter));
        }
    }
}

/// Opaque handle for iterating over objects within a single message.
pub struct TgmObjectIter {
    inner: tensogram::ObjectIter,
    /// Global metadata parsed from the message header, cloned into each
    /// yielded `TgmMessage` to preserve the original version and extra fields.
    global_metadata: GlobalMetadata,
}

/// Create an object iterator from raw message bytes.
///
/// Parses metadata once, then decodes each object on demand when
/// `tgm_object_iter_next` is called. The global metadata from the
/// original message is preserved in each yielded `TgmMessage`.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_object_iter_create(
    buf: *const u8,
    buf_len: usize,
    verify_hash: i32,
    native_byte_order: i32,
    out: *mut *mut TgmObjectIter,
) -> TgmError {
    if buf.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let data = unsafe { slice::from_raw_parts(buf, buf_len) };
    let options = DecodeOptions {
        verify_hash: verify_hash != 0,
        native_byte_order: native_byte_order != 0,
        ..Default::default()
    };

    // Parse global metadata from the message header so we can attach it to
    // each yielded TgmMessage instead of fabricating a default.
    let global_metadata = decode_metadata(data).unwrap_or_default();

    match tensogram::objects(data, options) {
        Ok(inner) => {
            let iter = Box::new(TgmObjectIter {
                inner,
                global_metadata,
            });
            unsafe {
                *out = Box::into_raw(iter);
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

/// Advance the object iterator. On success, fills `out` with a `TgmMessage`
/// handle containing exactly one decoded object (the next in sequence).
///
/// Returns `TgmError::Ok` when an object is available, `TgmError::EndOfIter`
/// when iteration is exhausted. Free each yielded `TgmMessage` with
/// `tgm_message_free`.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_object_iter_next(
    iter: *mut TgmObjectIter,
    out: *mut *mut TgmMessage,
) -> TgmError {
    if iter.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let it = unsafe { &mut *iter };
    match it.inner.next() {
        None => TgmError::EndOfIter,
        Some(Err(e)) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
        Some(Ok((descriptor, data))) => {
            let global_metadata = it.global_metadata.clone();
            let objects = vec![(descriptor, data)];
            let caches = build_message_caches(&objects);
            let msg = Box::new(TgmMessage {
                global_metadata,
                objects,
                dtype_strings: caches.dtype_strings,
                type_strings: caches.type_strings,
                byte_order_strings: caches.byte_order_strings,
                filter_strings: caches.filter_strings,
                compression_strings: caches.compression_strings,
                encoding_strings: caches.encoding_strings,
                hash_type_strings: caches.hash_type_strings,
                hash_value_strings: caches.hash_value_strings,
            });
            unsafe {
                *out = Box::into_raw(msg);
            }
            TgmError::Ok
        }
    }
}

/// Free an object iterator handle.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_object_iter_free(iter: *mut TgmObjectIter) {
    if !iter.is_null() {
        unsafe {
            drop(Box::from_raw(iter));
        }
    }
}

// ---------------------------------------------------------------------------
// Error code to string
// ---------------------------------------------------------------------------

/// Convert an error code to a human-readable string.
/// Returns a static string (always valid, never NULL).
///
/// Accepts a raw integer and matches by value so that invalid discriminants
/// from C callers do not trigger undefined behaviour in Rust.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_error_string(err: TgmError) -> *const c_char {
    // Convert to integer for safe matching — C callers may pass invalid values.
    let code = err as i32;
    let s: &[u8] = match code {
        0 => b"ok\0",
        1 => b"framing error\0",
        2 => b"metadata error\0",
        3 => b"encoding error\0",
        4 => b"compression error\0",
        5 => b"object error\0",
        6 => b"I/O error\0",
        7 => b"hash mismatch\0",
        8 => b"invalid argument\0",
        9 => b"end of iteration\0",
        10 => b"remote error\0",
        _ => b"unknown error\0",
    };
    s.as_ptr() as *const c_char
}

// ---------------------------------------------------------------------------
// Hash utilities
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Unit tests for metadata lookup helpers and JSON parsing
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn make_meta(
        base: Vec<BTreeMap<String, ciborium::Value>>,
        extra: BTreeMap<String, ciborium::Value>,
    ) -> GlobalMetadata {
        GlobalMetadata {
            version: 3,
            base,
            extra,
            ..Default::default()
        }
    }

    // ── lookup_cbor_value ─────────────────────────────────────────────

    #[test]
    fn lookup_cbor_empty_key() {
        let meta = make_meta(vec![], BTreeMap::new());
        assert!(lookup_cbor_value(&meta, "").is_none());
    }

    #[test]
    fn lookup_cbor_dot_only() {
        let meta = make_meta(vec![], BTreeMap::new());
        assert!(lookup_cbor_value(&meta, ".").is_none());
    }

    #[test]
    fn lookup_cbor_version_returns_none() {
        // version is handled by tgm_metadata_version, not lookup_cbor_value
        let meta = make_meta(vec![], BTreeMap::new());
        assert!(lookup_cbor_value(&meta, "version").is_none());
    }

    #[test]
    fn lookup_cbor_base_match() {
        let mut entry = BTreeMap::new();
        entry.insert("centre".into(), ciborium::Value::Text("ecmwf".into()));
        let meta = make_meta(vec![entry], BTreeMap::new());
        let val = lookup_cbor_value(&meta, "centre");
        assert!(matches!(val, Some(ciborium::Value::Text(s)) if s == "ecmwf"));
    }

    #[test]
    fn lookup_cbor_extra_fallback() {
        // Key not in base → found in extra
        let mut extra = BTreeMap::new();
        extra.insert("source".into(), ciborium::Value::Text("test".into()));
        let meta = make_meta(vec![], extra);
        let val = lookup_cbor_value(&meta, "source");
        assert!(matches!(val, Some(ciborium::Value::Text(s)) if s == "test"));
    }

    #[test]
    fn lookup_cbor_no_match() {
        let meta = make_meta(vec![], BTreeMap::new());
        assert!(lookup_cbor_value(&meta, "nonexistent").is_none());
    }

    #[test]
    fn lookup_cbor_reserved_skipped() {
        let mut entry = BTreeMap::new();
        entry.insert(
            "_reserved_".into(),
            ciborium::Value::Map(vec![(
                ciborium::Value::Text("tensor".into()),
                ciborium::Value::Text("internal".into()),
            )]),
        );
        entry.insert("param".into(), ciborium::Value::Text("2t".into()));
        let meta = make_meta(vec![entry], BTreeMap::new());
        // _reserved_ path should be skipped
        assert!(lookup_cbor_value(&meta, "_reserved_.tensor").is_none());
        // Regular key should still be found
        assert!(lookup_cbor_value(&meta, "param").is_some());
    }

    #[test]
    fn lookup_cbor_extra_prefix() {
        let mut extra = BTreeMap::new();
        extra.insert("custom".into(), ciborium::Value::Text("val".into()));
        let meta = make_meta(vec![], extra);
        // _extra_.custom should resolve directly in extra
        let val = lookup_cbor_value(&meta, "_extra_.custom");
        assert!(matches!(val, Some(ciborium::Value::Text(s)) if s == "val"));
    }

    #[test]
    fn lookup_cbor_extra_alias_prefix() {
        let mut extra = BTreeMap::new();
        extra.insert("custom".into(), ciborium::Value::Text("val".into()));
        let meta = make_meta(vec![], extra);
        // extra.custom should also resolve in extra
        let val = lookup_cbor_value(&meta, "extra.custom");
        assert!(matches!(val, Some(ciborium::Value::Text(s)) if s == "val"));
    }

    #[test]
    fn lookup_cbor_extra_prefix_alone_returns_none() {
        let meta = make_meta(vec![], BTreeMap::new());
        // Bare "_extra_" without subkey returns None
        assert!(lookup_cbor_value(&meta, "_extra_").is_none());
        assert!(lookup_cbor_value(&meta, "extra").is_none());
    }

    #[test]
    fn lookup_cbor_base_wins_over_extra() {
        let mut entry = BTreeMap::new();
        entry.insert("shared".into(), ciborium::Value::Text("from_base".into()));
        let mut extra = BTreeMap::new();
        extra.insert("shared".into(), ciborium::Value::Text("from_extra".into()));
        let meta = make_meta(vec![entry], extra);
        let val = lookup_cbor_value(&meta, "shared");
        assert!(matches!(val, Some(ciborium::Value::Text(s)) if s == "from_base"));
    }

    #[test]
    fn lookup_cbor_deeply_nested() {
        let e_val = ciborium::Value::Map(vec![(
            ciborium::Value::Text("e".into()),
            ciborium::Value::Text("deep".into()),
        )]);
        let d_val = ciborium::Value::Map(vec![(ciborium::Value::Text("d".into()), e_val)]);
        let c_val = ciborium::Value::Map(vec![(ciborium::Value::Text("c".into()), d_val)]);
        let b_val = ciborium::Value::Map(vec![(ciborium::Value::Text("b".into()), c_val)]);
        let mut entry = BTreeMap::new();
        entry.insert("a".into(), b_val);
        let meta = make_meta(vec![entry], BTreeMap::new());
        let val = lookup_cbor_value(&meta, "a.b.c.d.e");
        assert!(matches!(val, Some(ciborium::Value::Text(s)) if s == "deep"));
    }

    #[test]
    fn lookup_cbor_multi_base_first_match() {
        let mut entry0 = BTreeMap::new();
        entry0.insert("param".into(), ciborium::Value::Text("2t".into()));
        let mut entry1 = BTreeMap::new();
        entry1.insert("param".into(), ciborium::Value::Text("msl".into()));
        let meta = make_meta(vec![entry0, entry1], BTreeMap::new());
        let val = lookup_cbor_value(&meta, "param");
        assert!(matches!(val, Some(ciborium::Value::Text(s)) if s == "2t"));
    }

    // ── resolve_cbor_path ─────────────────────────────────────────────

    #[test]
    fn resolve_cbor_path_empty_remaining() {
        let value = ciborium::Value::Text("hello".into());
        assert_eq!(resolve_cbor_path(&value, &[]), Some(&value));
    }

    #[test]
    fn resolve_cbor_path_non_map_with_remaining() {
        let value = ciborium::Value::Text("hello".into());
        assert!(resolve_cbor_path(&value, &["key"]).is_none());
    }

    #[test]
    fn resolve_cbor_path_map_missing_key() {
        let value = ciborium::Value::Map(vec![(
            ciborium::Value::Text("a".into()),
            ciborium::Value::Text("b".into()),
        )]);
        assert!(resolve_cbor_path(&value, &["missing"]).is_none());
    }

    // ── lookup_string_key ──

    #[test]
    fn lookup_string_key_version() {
        let meta = make_meta(vec![], BTreeMap::new());
        assert_eq!(lookup_string_key(&meta, "version"), Some("3".into()));
    }

    #[test]
    fn lookup_string_key_empty() {
        let meta = make_meta(vec![], BTreeMap::new());
        assert!(lookup_string_key(&meta, "").is_none());
    }

    #[test]
    fn lookup_string_key_integer_value() {
        let mut entry = BTreeMap::new();
        entry.insert("count".into(), ciborium::Value::Integer(42.into()));
        let meta = make_meta(vec![entry], BTreeMap::new());
        assert_eq!(lookup_string_key(&meta, "count"), Some("42".into()));
    }

    #[test]
    fn lookup_string_key_float_value() {
        let mut extra = BTreeMap::new();
        extra.insert("temperature".into(), ciborium::Value::Float(98.6));
        let meta = make_meta(vec![], extra);
        assert_eq!(lookup_string_key(&meta, "temperature"), Some("98.6".into()));
    }

    #[test]
    fn lookup_string_key_bool_value() {
        let mut extra = BTreeMap::new();
        extra.insert("flag".into(), ciborium::Value::Bool(true));
        let meta = make_meta(vec![], extra);
        assert_eq!(lookup_string_key(&meta, "flag"), Some("true".into()));
    }

    #[test]
    fn lookup_string_key_null_returns_none() {
        let mut extra = BTreeMap::new();
        extra.insert("nothing".into(), ciborium::Value::Null);
        let meta = make_meta(vec![], extra);
        // Null is not a string/int/float/bool, so returns None
        assert!(lookup_string_key(&meta, "nothing").is_none());
    }

    // ── lookup_int_key ──

    #[test]
    fn lookup_int_key_version() {
        let meta = make_meta(vec![], BTreeMap::new());
        assert_eq!(lookup_int_key(&meta, "version"), Some(3));
    }

    #[test]
    fn lookup_int_key_non_integer() {
        let mut extra = BTreeMap::new();
        extra.insert("str".into(), ciborium::Value::Text("not_int".into()));
        let meta = make_meta(vec![], extra);
        assert!(lookup_int_key(&meta, "str").is_none());
    }

    // ── lookup_float_key ──

    #[test]
    fn lookup_float_key_float() {
        let mut extra = BTreeMap::new();
        extra.insert("val".into(), ciborium::Value::Float(98.6));
        let meta = make_meta(vec![], extra);
        assert_eq!(lookup_float_key(&meta, "val"), Some(98.6));
    }

    #[test]
    fn lookup_float_key_integer_coercion() {
        let mut extra = BTreeMap::new();
        extra.insert("count".into(), ciborium::Value::Integer(42.into()));
        let meta = make_meta(vec![], extra);
        assert_eq!(lookup_float_key(&meta, "count"), Some(42.0));
    }

    #[test]
    fn lookup_float_key_non_numeric() {
        let mut extra = BTreeMap::new();
        extra.insert("str".into(), ciborium::Value::Text("hello".into()));
        let meta = make_meta(vec![], extra);
        assert!(lookup_float_key(&meta, "str").is_none());
    }

    // ── parse_encode_json ──

    #[test]
    fn parse_encode_json_with_base() {
        let json = r#"{"version":3,"base":[{"mars":{"param":"2t"}}],"descriptors":[]}"#;
        let (gm, descs) = parse_encode_json(json).unwrap();
        assert_eq!(gm.version, 3);
        assert_eq!(gm.base.len(), 1);
        assert!(gm.base[0].contains_key("mars"));
        assert!(descs.is_empty());
    }

    #[test]
    fn parse_encode_json_without_base() {
        let json = r#"{"version":3,"descriptors":[]}"#;
        let (gm, _) = parse_encode_json(json).unwrap();
        assert!(gm.base.is_empty());
    }

    #[test]
    fn parse_encode_json_reserved_in_base_rejected() {
        let json = r#"{"version":3,"base":[{"_reserved_":{"tensor":{}}}],"descriptors":[]}"#;
        let result = parse_encode_json(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("_reserved_"));
    }

    #[test]
    fn parse_encode_json_extra_keys() {
        let json = r#"{"version":3,"descriptors":[],"source":"test","count":42}"#;
        let (gm, _) = parse_encode_json(json).unwrap();
        assert!(gm.extra.contains_key("source"));
        assert!(gm.extra.contains_key("count"));
    }

    // ── parse_streaming_metadata_json ──

    #[test]
    fn parse_streaming_json_with_base() {
        let json = r#"{"version":3,"base":[{"mars":{"param":"2t"}}]}"#;
        let gm = parse_streaming_metadata_json(json).unwrap();
        assert_eq!(gm.version, 3);
        assert_eq!(gm.base.len(), 1);
    }

    #[test]
    fn parse_streaming_json_reserved_rejected() {
        let json = r#"{"version":3,"base":[{"_reserved_":{"tensor":{}}}]}"#;
        let result = parse_streaming_metadata_json(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("_reserved_"));
    }

    #[test]
    fn parse_streaming_json_no_base() {
        let json = r#"{"version":3,"source":"stream"}"#;
        let gm = parse_streaming_metadata_json(json).unwrap();
        assert!(gm.base.is_empty());
        assert!(gm.extra.contains_key("source"));
    }

    #[test]
    fn parse_streaming_json_invalid_json() {
        assert!(parse_streaming_metadata_json("not json").is_err());
    }

    #[test]
    fn parse_encode_json_invalid_json() {
        assert!(parse_encode_json("not json").is_err());
    }

    // ── json_to_cbor ──

    #[test]
    fn json_to_cbor_null() {
        assert_eq!(json_to_cbor(serde_json::Value::Null), ciborium::Value::Null);
    }

    #[test]
    fn json_to_cbor_bool() {
        assert_eq!(
            json_to_cbor(serde_json::Value::Bool(true)),
            ciborium::Value::Bool(true)
        );
    }

    #[test]
    fn json_to_cbor_integer() {
        let val = serde_json::json!(42);
        let cbor = json_to_cbor(val);
        assert!(matches!(cbor, ciborium::Value::Integer(_)));
    }

    #[test]
    fn json_to_cbor_float() {
        let val = serde_json::json!(98.6);
        let cbor = json_to_cbor(val);
        assert!(matches!(cbor, ciborium::Value::Float(_)));
    }

    #[test]
    fn json_to_cbor_string() {
        let val = serde_json::json!("hello");
        let cbor = json_to_cbor(val);
        assert!(matches!(cbor, ciborium::Value::Text(s) if s == "hello"));
    }

    #[test]
    fn json_to_cbor_array() {
        let val = serde_json::json!([1, 2, 3]);
        let cbor = json_to_cbor(val);
        assert!(matches!(cbor, ciborium::Value::Array(_)));
    }

    #[test]
    fn json_to_cbor_object() {
        let val = serde_json::json!({"key": "value"});
        let cbor = json_to_cbor(val);
        assert!(matches!(cbor, ciborium::Value::Map(_)));
    }

    #[test]
    fn json_to_cbor_u64_fallback_to_float() {
        // A number that is not i64 but is u64 → falls back to float
        // (JSON numbers outside i64 range)
        let val = serde_json::json!(18446744073709551615u64);
        let cbor = json_to_cbor(val);
        // This should be either Integer or Float depending on serde_json parsing
        assert!(!matches!(cbor, ciborium::Value::Null));
    }

    // ── resolve helpers ──

    #[test]
    fn resolve_in_btree_skip_reserved_blocks_reserved() {
        let mut map = BTreeMap::new();
        map.insert("_reserved_".into(), ciborium::Value::Text("secret".into()));
        assert!(resolve_in_btree_skip_reserved(&map, &["_reserved_"]).is_none());
    }

    #[test]
    fn resolve_in_btree_empty_parts() {
        let map = BTreeMap::new();
        assert!(resolve_in_btree(&map, &[]).is_none());
    }

    #[test]
    fn resolve_in_btree_skip_reserved_empty_parts() {
        let map = BTreeMap::new();
        assert!(resolve_in_btree_skip_reserved(&map, &[]).is_none());
    }

    // ── validate FFI ──

    #[test]
    fn parse_validate_options_default() {
        let opts = match super::parse_validate_options(ptr::null(), 0) {
            Ok(opts) => opts,
            Err((_code, msg)) => panic!("expected default options, got error: {msg}"),
        };
        assert_eq!(opts.max_level, ValidationLevel::Integrity);
        assert!(!opts.check_canonical);
        assert!(!opts.checksum_only);
    }

    #[test]
    fn parse_validate_options_quick() {
        let level = CString::new("quick").unwrap();
        let opts = match super::parse_validate_options(level.as_ptr(), 0) {
            Ok(opts) => opts,
            Err((_code, msg)) => panic!("expected quick options, got error: {msg}"),
        };
        assert_eq!(opts.max_level, ValidationLevel::Structure);
    }

    #[test]
    fn parse_validate_options_full_canonical() {
        let level = CString::new("full").unwrap();
        let opts = match super::parse_validate_options(level.as_ptr(), 1) {
            Ok(opts) => opts,
            Err((_code, msg)) => panic!("expected full options, got error: {msg}"),
        };
        assert_eq!(opts.max_level, ValidationLevel::Fidelity);
        assert!(opts.check_canonical);
    }

    #[test]
    fn parse_validate_options_unknown_level() {
        let level = CString::new("bogus").unwrap();
        let result = super::parse_validate_options(level.as_ptr(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn parse_validate_options_checksum() {
        let level = CString::new("checksum").unwrap();
        let opts = match super::parse_validate_options(level.as_ptr(), 0) {
            Ok(opts) => opts,
            Err((_code, msg)) => panic!("expected checksum options, got error: {msg}"),
        };
        assert_eq!(opts.max_level, ValidationLevel::Integrity);
        assert!(opts.checksum_only);
    }

    // ── tgm_validate end-to-end ──

    fn encode_test_message() -> Vec<u8> {
        let meta = GlobalMetadata::default();
        let desc = DataObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 1,
            shape: vec![4],
            strides: vec![1],
            dtype: tensogram::Dtype::Float32,
            byte_order: tensogram::ByteOrder::native(),
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            masks: None,
        };
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        tensogram::encode(&meta, &[(&desc, data.as_slice())], &Default::default()).unwrap()
    }

    #[test]
    fn tgm_validate_valid_message() {
        let msg = encode_test_message();
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_validate(msg.as_ptr(), msg.len(), ptr::null(), 0, &mut out);
        assert!(matches!(err, super::TgmError::Ok));
        assert!(!out.data.is_null());
        assert!(out.len > 0);
        let json_str =
            unsafe { std::str::from_utf8(std::slice::from_raw_parts(out.data, out.len)).unwrap() };
        assert!(json_str.contains("\"issues\":[]"));
        assert!(json_str.contains("\"object_count\":1"));
        super::tgm_bytes_free(out);
    }

    #[test]
    fn tgm_validate_empty_buffer() {
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_validate(ptr::null(), 0, ptr::null(), 0, &mut out);
        assert!(matches!(err, super::TgmError::Ok));
        let json_str =
            unsafe { std::str::from_utf8(std::slice::from_raw_parts(out.data, out.len)).unwrap() };
        assert!(json_str.contains("\"buffer_too_short\""));
        super::tgm_bytes_free(out);
    }

    #[test]
    fn tgm_validate_invalid_level() {
        let msg = encode_test_message();
        let level = CString::new("bogus").unwrap();
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_validate(msg.as_ptr(), msg.len(), level.as_ptr(), 0, &mut out);
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn tgm_validate_null_out() {
        let msg = encode_test_message();
        let err = super::tgm_validate(msg.as_ptr(), msg.len(), ptr::null(), 0, ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn tgm_validate_file_nonexistent() {
        let path = CString::new("/nonexistent/path/to/file.tgm").unwrap();
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_validate_file(path.as_ptr(), ptr::null(), 0, &mut out);
        assert!(matches!(err, super::TgmError::Io));
    }

    #[test]
    fn tgm_validate_file_null_out() {
        let path = CString::new("/tmp/dummy.tgm").unwrap();
        let err = super::tgm_validate_file(path.as_ptr(), ptr::null(), 0, ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn tgm_validate_file_invalid_level() {
        let path = CString::new("/tmp/dummy.tgm").unwrap();
        let level = CString::new("bogus").unwrap();
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_validate_file(path.as_ptr(), level.as_ptr(), 0, &mut out);
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    // =====================================================================
    // FFI round-trip tests — exercise #[no_mangle] extern "C" functions
    // =====================================================================

    /// Helper: build a JSON metadata string and raw data for a single float32
    /// tensor, encode via `tgm_encode`, and return the encoded bytes.
    fn ffi_encode_single_f32_tensor(values: &[f32], extra_json: &str) -> Vec<u8> {
        let shape_str = format!("[{}]", values.len());
        let json = format!(
            r#"{{"version":3,"descriptors":[{{"type":"ntensor","ndim":1,"shape":{shape},"strides":[1],"dtype":"float32","byte_order":"{bo}","encoding":"none","filter":"none","compression":"none"}}]{extra}}}"#,
            shape = shape_str,
            bo = if cfg!(target_endian = "little") {
                "little"
            } else {
                "big"
            },
            extra = if extra_json.is_empty() {
                String::new()
            } else {
                format!(",{extra_json}")
            },
        );

        let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let c_json = CString::new(json).unwrap();
        let data_ptr: *const u8 = data.as_ptr();
        let data_len: usize = data.len();

        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };

        let err = super::tgm_encode(
            c_json.as_ptr(),
            &data_ptr as *const *const u8,
            &data_len as *const usize,
            1,
            ptr::null(), // no hash
            0,           // threads
            &mut out,
        );
        assert!(matches!(err, super::TgmError::Ok), "tgm_encode failed");
        assert!(!out.data.is_null());
        assert!(out.len > 0);

        let encoded = unsafe { slice::from_raw_parts(out.data, out.len) }.to_vec();
        super::tgm_bytes_free(out);
        encoded
    }

    /// Helper: encode with hash enabled.
    fn ffi_encode_with_hash(values: &[f32]) -> Vec<u8> {
        let shape_str = format!("[{}]", values.len());
        let json = format!(
            r#"{{"version":3,"descriptors":[{{"type":"ntensor","ndim":1,"shape":{shape},"strides":[1],"dtype":"float32","byte_order":"{bo}","encoding":"none","filter":"none","compression":"none"}}]}}"#,
            shape = shape_str,
            bo = if cfg!(target_endian = "little") {
                "little"
            } else {
                "big"
            },
        );

        let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let c_json = CString::new(json).unwrap();
        let hash_algo = CString::new("xxh3").unwrap();
        let data_ptr: *const u8 = data.as_ptr();
        let data_len: usize = data.len();

        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };

        let err = super::tgm_encode(
            c_json.as_ptr(),
            &data_ptr as *const *const u8,
            &data_len as *const usize,
            1,
            hash_algo.as_ptr(),
            0,
            &mut out,
        );
        assert!(
            matches!(err, super::TgmError::Ok),
            "tgm_encode with hash failed"
        );

        let encoded = unsafe { slice::from_raw_parts(out.data, out.len) }.to_vec();
        super::tgm_bytes_free(out);
        encoded
    }

    // ── tgm_encode / tgm_decode round-trip ──

    #[test]
    fn ffi_encode_decode_round_trip() {
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let encoded = ffi_encode_single_f32_tensor(&values, "");

        // Decode
        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_decode(
            encoded.as_ptr(),
            encoded.len(),
            0, // no hash verify
            0, // no native byte order rewrite
            0, // threads
            &mut msg,
        );
        assert!(matches!(err, super::TgmError::Ok));
        assert!(!msg.is_null());

        // Message-level accessors
        assert_eq!(super::tgm_message_version(msg), 3);
        assert_eq!(super::tgm_message_num_objects(msg), 1);
        assert_eq!(super::tgm_message_num_decoded(msg), 1);

        // Object-level accessors
        assert_eq!(super::tgm_object_ndim(msg, 0), 1);

        let shape_ptr = super::tgm_object_shape(msg, 0);
        assert!(!shape_ptr.is_null());
        assert_eq!(unsafe { *shape_ptr }, 4);

        let strides_ptr = super::tgm_object_strides(msg, 0);
        assert!(!strides_ptr.is_null());
        assert_eq!(unsafe { *strides_ptr }, 1);

        // dtype string
        let dtype_ptr = super::tgm_object_dtype(msg, 0);
        assert!(!dtype_ptr.is_null());
        let dtype_str = unsafe { CStr::from_ptr(dtype_ptr) }.to_str().unwrap();
        assert_eq!(dtype_str, "float32");

        // type string
        let type_ptr = super::tgm_object_type(msg, 0);
        assert!(!type_ptr.is_null());
        let type_str = unsafe { CStr::from_ptr(type_ptr) }.to_str().unwrap();
        assert_eq!(type_str, "ntensor");

        // byte_order string
        let bo_ptr = super::tgm_object_byte_order(msg, 0);
        assert!(!bo_ptr.is_null());
        let bo_str = unsafe { CStr::from_ptr(bo_ptr) }.to_str().unwrap();
        assert!(bo_str == "little" || bo_str == "big");

        // filter string
        let filter_ptr = super::tgm_object_filter(msg, 0);
        assert!(!filter_ptr.is_null());
        let filter_str = unsafe { CStr::from_ptr(filter_ptr) }.to_str().unwrap();
        assert_eq!(filter_str, "none");

        // compression string
        let comp_ptr = super::tgm_object_compression(msg, 0);
        assert!(!comp_ptr.is_null());
        let comp_str = unsafe { CStr::from_ptr(comp_ptr) }.to_str().unwrap();
        assert_eq!(comp_str, "none");

        // encoding string
        let enc_ptr = super::tgm_payload_encoding(msg, 0);
        assert!(!enc_ptr.is_null());
        let enc_str = unsafe { CStr::from_ptr(enc_ptr) }.to_str().unwrap();
        assert_eq!(enc_str, "none");

        // decoded data
        let mut data_len: usize = 0;
        let data_ptr = super::tgm_object_data(msg, 0, &mut data_len);
        assert!(!data_ptr.is_null());
        assert_eq!(data_len, 16); // 4 × 4 bytes

        let decoded_bytes = unsafe { slice::from_raw_parts(data_ptr, data_len) };
        let decoded_values: Vec<f32> = decoded_bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(decoded_values, values);

        super::tgm_message_free(msg);
    }

    #[ignore = "v3: hash moved to frame footer — re-enable in phase 6"]
    #[test]
    fn ffi_encode_decode_with_hash() {
        let values = [10.0f32, 20.0, 30.0];
        let encoded = ffi_encode_with_hash(&values);

        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_decode(
            encoded.as_ptr(),
            encoded.len(),
            1, // verify hash
            0,
            0, // threads
            &mut msg,
        );
        assert!(matches!(err, super::TgmError::Ok));

        // Hash should be present
        assert_eq!(super::tgm_payload_has_hash(msg, 0), 1);

        let ht_ptr = super::tgm_object_hash_type(msg, 0);
        assert!(!ht_ptr.is_null());
        let ht_str = unsafe { CStr::from_ptr(ht_ptr) }.to_str().unwrap();
        assert_eq!(ht_str, "xxh3");

        let hv_ptr = super::tgm_object_hash_value(msg, 0);
        assert!(!hv_ptr.is_null());
        let hv_str = unsafe { CStr::from_ptr(hv_ptr) }.to_str().unwrap();
        assert!(!hv_str.is_empty());

        super::tgm_message_free(msg);
    }

    #[test]
    fn ffi_encode_decode_no_hash() {
        let values = [5.0f32];
        let encoded = ffi_encode_single_f32_tensor(&values, "");

        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_decode(encoded.as_ptr(), encoded.len(), 0, 0, 0, &mut msg);
        assert!(matches!(err, super::TgmError::Ok));

        assert_eq!(super::tgm_payload_has_hash(msg, 0), 0);
        assert!(super::tgm_object_hash_type(msg, 0).is_null());
        assert!(super::tgm_object_hash_value(msg, 0).is_null());

        super::tgm_message_free(msg);
    }

    #[test]
    fn ffi_encode_with_extra_metadata() {
        let values = [1.0f32, 2.0];
        let encoded = ffi_encode_single_f32_tensor(&values, r#""source":"test_source","count":42"#);

        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_decode(encoded.as_ptr(), encoded.len(), 0, 0, 0, &mut msg);
        assert!(matches!(err, super::TgmError::Ok));

        // Extract metadata from decoded message
        let mut meta: *mut super::TgmMetadata = ptr::null_mut();
        let err = super::tgm_message_metadata(msg, &mut meta);
        assert!(matches!(err, super::TgmError::Ok));

        let key = CString::new("source").unwrap();
        let val_ptr = super::tgm_metadata_get_string(meta, key.as_ptr());
        assert!(!val_ptr.is_null());
        let val_str = unsafe { CStr::from_ptr(val_ptr) }.to_str().unwrap();
        assert_eq!(val_str, "test_source");

        let key_count = CString::new("count").unwrap();
        let val_int = super::tgm_metadata_get_int(meta, key_count.as_ptr(), -1);
        assert_eq!(val_int, 42);

        super::tgm_metadata_free(meta);
        super::tgm_message_free(msg);
    }

    // ── tgm_encode null/error paths ──

    #[test]
    fn ffi_encode_null_json() {
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_encode(
            ptr::null(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            &mut out,
        );
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn ffi_encode_null_out() {
        let json = CString::new(r#"{"version":3,"descriptors":[]}"#).unwrap();
        let err = super::tgm_encode(
            json.as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            ptr::null_mut(),
        );
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn ffi_encode_descriptor_count_mismatch() {
        // JSON says 0 descriptors, but num_objects = 1
        let json = CString::new(r#"{"version":3,"descriptors":[]}"#).unwrap();
        let data: [u8; 4] = [0; 4];
        let data_ptr: *const u8 = data.as_ptr();
        let data_len: usize = 4;
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_encode(
            json.as_ptr(),
            &data_ptr as *const *const u8,
            &data_len as *const usize,
            1, // mismatch!
            ptr::null(),
            0, // threads
            &mut out,
        );
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn ffi_encode_invalid_json() {
        let json = CString::new("not valid json").unwrap();
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_encode(
            json.as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            &mut out,
        );
        assert!(matches!(err, super::TgmError::Metadata));
    }

    // ── tgm_decode null/error paths ──

    #[test]
    fn ffi_decode_null_buf() {
        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_decode(ptr::null(), 0, 0, 0, 0, &mut msg);
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn ffi_decode_null_out() {
        let data = [0u8; 10];
        let err = super::tgm_decode(data.as_ptr(), data.len(), 0, 0, 0, ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn ffi_decode_garbage_data() {
        let data = [0u8; 10];
        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_decode(data.as_ptr(), data.len(), 0, 0, 0, &mut msg);
        // Should fail with a framing or other error
        assert!(!matches!(err, super::TgmError::Ok));
    }

    // ── tgm_decode_metadata round-trip ──

    #[test]
    fn ffi_decode_metadata_round_trip() {
        let values = [1.0f32, 2.0];
        let encoded = ffi_encode_single_f32_tensor(&values, r#""source":"meta_test""#);

        let mut meta: *mut super::TgmMetadata = ptr::null_mut();
        let err = super::tgm_decode_metadata(encoded.as_ptr(), encoded.len(), &mut meta);
        assert!(matches!(err, super::TgmError::Ok));
        assert!(!meta.is_null());

        // Version
        assert_eq!(super::tgm_metadata_version(meta), 3);

        // num_objects
        assert_eq!(super::tgm_metadata_num_objects(meta), 1);

        // String lookup
        let key = CString::new("source").unwrap();
        let val_ptr = super::tgm_metadata_get_string(meta, key.as_ptr());
        assert!(!val_ptr.is_null());
        let val_str = unsafe { CStr::from_ptr(val_ptr) }.to_str().unwrap();
        assert_eq!(val_str, "meta_test");

        // Missing key returns null
        let bad_key = CString::new("nonexistent").unwrap();
        assert!(super::tgm_metadata_get_string(meta, bad_key.as_ptr()).is_null());

        // Int with default
        let bad_key2 = CString::new("missing_int").unwrap();
        assert_eq!(
            super::tgm_metadata_get_int(meta, bad_key2.as_ptr(), -999),
            -999
        );

        // Float with default
        let bad_key3 = CString::new("missing_float").unwrap();
        let fval = super::tgm_metadata_get_float(meta, bad_key3.as_ptr(), 3.25);
        assert!((fval - 3.25).abs() < f64::EPSILON);

        super::tgm_metadata_free(meta);
    }

    #[test]
    fn ffi_decode_metadata_null_args() {
        let mut meta: *mut super::TgmMetadata = ptr::null_mut();
        let err = super::tgm_decode_metadata(ptr::null(), 0, &mut meta);
        assert!(matches!(err, super::TgmError::InvalidArg));

        let data = [0u8; 10];
        let err = super::tgm_decode_metadata(data.as_ptr(), data.len(), ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    // ── tgm_metadata null pointer safety ──

    #[test]
    fn ffi_metadata_accessors_null_handle() {
        assert_eq!(super::tgm_metadata_version(ptr::null()), 0);
        assert_eq!(super::tgm_metadata_num_objects(ptr::null()), 0);
        assert!(super::tgm_metadata_get_string(ptr::null(), ptr::null()).is_null());
        assert_eq!(
            super::tgm_metadata_get_int(ptr::null(), ptr::null(), -1),
            -1
        );
        assert_eq!(
            super::tgm_metadata_get_float(ptr::null(), ptr::null(), 1.5),
            1.5
        );
    }

    #[test]
    fn ffi_metadata_get_string_null_key() {
        let encoded = ffi_encode_single_f32_tensor(&[1.0f32], "");
        let mut meta: *mut super::TgmMetadata = ptr::null_mut();
        let err = super::tgm_decode_metadata(encoded.as_ptr(), encoded.len(), &mut meta);
        assert!(matches!(err, super::TgmError::Ok));

        assert!(super::tgm_metadata_get_string(meta, ptr::null()).is_null());
        assert_eq!(super::tgm_metadata_get_int(meta, ptr::null(), -1), -1);
        assert_eq!(super::tgm_metadata_get_float(meta, ptr::null(), 1.5), 1.5);

        super::tgm_metadata_free(meta);
    }

    #[test]
    fn ffi_metadata_get_version_via_string() {
        let encoded = ffi_encode_single_f32_tensor(&[1.0f32], "");
        let mut meta: *mut super::TgmMetadata = ptr::null_mut();
        let err = super::tgm_decode_metadata(encoded.as_ptr(), encoded.len(), &mut meta);
        assert!(matches!(err, super::TgmError::Ok));

        let key = CString::new("version").unwrap();
        let val_ptr = super::tgm_metadata_get_string(meta, key.as_ptr());
        assert!(!val_ptr.is_null());
        let val_str = unsafe { CStr::from_ptr(val_ptr) }.to_str().unwrap();
        assert_eq!(val_str, "3");

        let ival = super::tgm_metadata_get_int(meta, key.as_ptr(), -1);
        assert_eq!(ival, 3);

        super::tgm_metadata_free(meta);
    }

    #[test]
    fn ffi_metadata_get_float_value() {
        let values = [1.0f32];
        let encoded = ffi_encode_single_f32_tensor(&values, r#""temperature":98.6"#);

        let mut meta: *mut super::TgmMetadata = ptr::null_mut();
        let err = super::tgm_decode_metadata(encoded.as_ptr(), encoded.len(), &mut meta);
        assert!(matches!(err, super::TgmError::Ok));

        let key = CString::new("temperature").unwrap();
        let fval = super::tgm_metadata_get_float(meta, key.as_ptr(), 0.0);
        assert!((fval - 98.6).abs() < 0.01);

        super::tgm_metadata_free(meta);
    }

    // ── tgm_decode_object ──

    #[test]
    fn ffi_decode_object_round_trip() {
        let values = [10.0f32, 20.0, 30.0, 40.0];
        let encoded = ffi_encode_single_f32_tensor(&values, "");

        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_decode_object(
            encoded.as_ptr(),
            encoded.len(),
            0, // index
            0, // verify hash
            0, // native byte order
            0, // threads
            &mut msg,
        );
        assert!(matches!(err, super::TgmError::Ok));
        assert!(!msg.is_null());

        // Single object in result
        assert_eq!(super::tgm_message_num_objects(msg), 1);
        assert_eq!(super::tgm_object_ndim(msg, 0), 1);

        let mut data_len: usize = 0;
        let data_ptr = super::tgm_object_data(msg, 0, &mut data_len);
        assert!(!data_ptr.is_null());
        let decoded_bytes = unsafe { slice::from_raw_parts(data_ptr, data_len) };
        let decoded_values: Vec<f32> = decoded_bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(decoded_values, values);

        super::tgm_message_free(msg);
    }

    #[test]
    fn ffi_decode_object_out_of_range() {
        let encoded = ffi_encode_single_f32_tensor(&[1.0f32], "");
        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_decode_object(
            encoded.as_ptr(),
            encoded.len(),
            999, // out of range
            0,
            0,
            0, // threads
            &mut msg,
        );
        assert!(!matches!(err, super::TgmError::Ok));
    }

    #[test]
    fn ffi_decode_object_null_args() {
        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_decode_object(ptr::null(), 0, 0, 0, 0, 0, &mut msg);
        assert!(matches!(err, super::TgmError::InvalidArg));

        let data = [0u8; 10];
        let err = super::tgm_decode_object(data.as_ptr(), data.len(), 0, 0, 0, 0, ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    // ── tgm_message_metadata ──

    #[test]
    fn ffi_message_metadata_null_args() {
        let err = super::tgm_message_metadata(ptr::null(), ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));

        let encoded = ffi_encode_single_f32_tensor(&[1.0f32], "");
        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_decode(encoded.as_ptr(), encoded.len(), 0, 0, 0, &mut msg);
        assert!(matches!(err, super::TgmError::Ok));

        let err = super::tgm_message_metadata(msg, ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));

        super::tgm_message_free(msg);
    }

    // ── tgm_message accessors with null msg ──

    #[test]
    fn ffi_message_accessors_null_handle() {
        assert_eq!(super::tgm_message_version(ptr::null()), 0);
        assert_eq!(super::tgm_message_num_objects(ptr::null()), 0);
        assert_eq!(super::tgm_message_num_decoded(ptr::null()), 0);
        assert_eq!(super::tgm_object_ndim(ptr::null(), 0), 0);
        assert!(super::tgm_object_shape(ptr::null(), 0).is_null());
        assert!(super::tgm_object_strides(ptr::null(), 0).is_null());
        assert!(super::tgm_object_dtype(ptr::null(), 0).is_null());
        assert!(super::tgm_object_type(ptr::null(), 0).is_null());
        assert!(super::tgm_object_byte_order(ptr::null(), 0).is_null());
        assert!(super::tgm_object_filter(ptr::null(), 0).is_null());
        assert!(super::tgm_object_compression(ptr::null(), 0).is_null());
        assert!(super::tgm_payload_encoding(ptr::null(), 0).is_null());
        assert_eq!(super::tgm_payload_has_hash(ptr::null(), 0), 0);
        assert!(super::tgm_object_hash_type(ptr::null(), 0).is_null());
        assert!(super::tgm_object_hash_value(ptr::null(), 0).is_null());

        let mut data_len: usize = 99;
        let data_ptr = super::tgm_object_data(ptr::null(), 0, &mut data_len);
        assert!(data_ptr.is_null());
        assert_eq!(data_len, 0);
    }

    // ── tgm_message accessors out-of-bounds index ──

    #[test]
    fn ffi_message_accessors_out_of_bounds() {
        let encoded = ffi_encode_single_f32_tensor(&[1.0f32], "");
        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_decode(encoded.as_ptr(), encoded.len(), 0, 0, 0, &mut msg);
        assert!(matches!(err, super::TgmError::Ok));

        // Index 1 does not exist (only index 0)
        assert_eq!(super::tgm_object_ndim(msg, 1), 0);
        assert!(super::tgm_object_shape(msg, 1).is_null());
        assert!(super::tgm_object_strides(msg, 1).is_null());
        assert!(super::tgm_object_dtype(msg, 1).is_null());
        assert!(super::tgm_object_type(msg, 1).is_null());
        assert!(super::tgm_object_byte_order(msg, 1).is_null());
        assert!(super::tgm_object_filter(msg, 1).is_null());
        assert!(super::tgm_object_compression(msg, 1).is_null());
        assert!(super::tgm_payload_encoding(msg, 1).is_null());
        assert_eq!(super::tgm_payload_has_hash(msg, 1), 0);
        assert!(super::tgm_object_hash_type(msg, 1).is_null());
        assert!(super::tgm_object_hash_value(msg, 1).is_null());

        let mut data_len: usize = 99;
        let data_ptr = super::tgm_object_data(msg, 1, &mut data_len);
        assert!(data_ptr.is_null());
        assert_eq!(data_len, 0);

        super::tgm_message_free(msg);
    }

    // ── tgm_object_data with null out_len ──

    #[test]
    fn ffi_object_data_null_out_len() {
        let encoded = ffi_encode_single_f32_tensor(&[1.0f32], "");
        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_decode(encoded.as_ptr(), encoded.len(), 0, 0, 0, &mut msg);
        assert!(matches!(err, super::TgmError::Ok));

        // null out_len should not crash
        let data_ptr = super::tgm_object_data(msg, 0, ptr::null_mut());
        assert!(!data_ptr.is_null());

        super::tgm_message_free(msg);
    }

    // ── tgm_decode_range ──

    #[test]
    fn ffi_decode_range_round_trip() {
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let encoded = ffi_encode_single_f32_tensor(&values, "");

        // Request elements [2..5) (3 elements)
        let range_offset: u64 = 2;
        let range_count: u64 = 3;
        let mut out_buf = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let mut out_count: usize = 0;

        let err = super::tgm_decode_range(
            encoded.as_ptr(),
            encoded.len(),
            0,
            &range_offset as *const u64,
            &range_count as *const u64,
            1,
            0, // no hash verify
            0, // no native byte order
            0, // threads
            1, // join
            &mut out_buf,
            &mut out_count,
        );
        assert!(matches!(err, super::TgmError::Ok));
        assert_eq!(out_count, 1);
        assert!(!out_buf.data.is_null());

        let decoded_bytes = unsafe { slice::from_raw_parts(out_buf.data, out_buf.len) };
        let decoded_values: Vec<f32> = decoded_bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(decoded_values, [3.0, 4.0, 5.0]);

        super::tgm_bytes_free(out_buf);
    }

    #[test]
    fn ffi_decode_range_split_mode() {
        let values = [10.0f32, 20.0, 30.0, 40.0];
        let encoded = ffi_encode_single_f32_tensor(&values, "");

        // Two ranges: [0..2), [2..4)
        let range_offsets: [u64; 2] = [0, 2];
        let range_counts: [u64; 2] = [2, 2];
        let mut out_bufs = [
            super::TgmBytes {
                data: ptr::null_mut(),
                len: 0,
            },
            super::TgmBytes {
                data: ptr::null_mut(),
                len: 0,
            },
        ];
        let mut out_count: usize = 0;

        let err = super::tgm_decode_range(
            encoded.as_ptr(),
            encoded.len(),
            0,
            range_offsets.as_ptr(),
            range_counts.as_ptr(),
            2,
            0,
            0,
            0, // threads
            0, // split mode (join=0)
            out_bufs.as_mut_ptr(),
            &mut out_count,
        );
        assert!(matches!(err, super::TgmError::Ok));
        assert_eq!(out_count, 2);

        // First range: [10.0, 20.0]
        let bytes0 = unsafe { slice::from_raw_parts(out_bufs[0].data, out_bufs[0].len) };
        let vals0: Vec<f32> = bytes0
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals0, [10.0, 20.0]);

        // Second range: [30.0, 40.0]
        let bytes1 = unsafe { slice::from_raw_parts(out_bufs[1].data, out_bufs[1].len) };
        let vals1: Vec<f32> = bytes1
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals1, [30.0, 40.0]);

        // TgmBytes is not Copy, so manually construct values for free
        super::tgm_bytes_free(super::TgmBytes {
            data: out_bufs[0].data,
            len: out_bufs[0].len,
        });
        super::tgm_bytes_free(super::TgmBytes {
            data: out_bufs[1].data,
            len: out_bufs[1].len,
        });
    }

    #[test]
    fn ffi_decode_range_null_args() {
        let mut out_buf = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let mut out_count: usize = 0;

        // null buf
        let err = super::tgm_decode_range(
            ptr::null(),
            0,
            0,
            ptr::null(),
            ptr::null(),
            0,
            0,
            0,
            0,
            0,
            &mut out_buf,
            &mut out_count,
        );
        assert!(matches!(err, super::TgmError::InvalidArg));

        // null out
        let data = [0u8; 10];
        let err = super::tgm_decode_range(
            data.as_ptr(),
            data.len(),
            0,
            ptr::null(),
            ptr::null(),
            0,
            0,
            0,
            0,
            0,
            ptr::null_mut(),
            &mut out_count,
        );
        assert!(matches!(err, super::TgmError::InvalidArg));

        // null out_count
        let err = super::tgm_decode_range(
            data.as_ptr(),
            data.len(),
            0,
            ptr::null(),
            ptr::null(),
            0,
            0,
            0,
            0,
            0,
            &mut out_buf,
            ptr::null_mut(),
        );
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn ffi_decode_range_null_ranges_with_nonzero_count() {
        let encoded = ffi_encode_single_f32_tensor(&[1.0f32], "");
        let mut out_buf = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let mut out_count: usize = 0;

        let err = super::tgm_decode_range(
            encoded.as_ptr(),
            encoded.len(),
            0,
            ptr::null(), // null ranges_offsets
            ptr::null(), // null ranges_counts
            1,           // but num_ranges > 0
            0,
            0,
            0, // threads
            0,
            &mut out_buf,
            &mut out_count,
        );
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    // ── tgm_scan ──

    #[test]
    fn ffi_scan_single_message() {
        let encoded = ffi_encode_single_f32_tensor(&[1.0f32, 2.0], "");

        let mut result: *mut super::TgmScanResult = ptr::null_mut();
        let err = super::tgm_scan(encoded.as_ptr(), encoded.len(), &mut result);
        assert!(matches!(err, super::TgmError::Ok));
        assert!(!result.is_null());

        assert_eq!(super::tgm_scan_count(result), 1);

        let entry = super::tgm_scan_entry(result, 0);
        assert_eq!(entry.offset, 0);
        assert_eq!(entry.length, encoded.len());

        // Out of bounds entry returns sentinel (offset=usize::MAX, length=0)
        // and sets tgm_last_error
        let bad = super::tgm_scan_entry(result, 999);
        assert_eq!(bad.offset, usize::MAX);
        assert_eq!(bad.length, 0);
        let err_ptr = super::tgm_last_error();
        assert!(!err_ptr.is_null());
        let err_str = unsafe { CStr::from_ptr(err_ptr) }.to_str().unwrap();
        assert!(
            err_str.contains("out of range"),
            "expected OOB error, got: {err_str}"
        );

        super::tgm_scan_free(result);
    }

    #[test]
    fn ffi_scan_null_args() {
        let mut result: *mut super::TgmScanResult = ptr::null_mut();
        let err = super::tgm_scan(ptr::null(), 0, &mut result);
        assert!(matches!(err, super::TgmError::InvalidArg));

        let data = [0u8; 10];
        let err = super::tgm_scan(data.as_ptr(), data.len(), ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn ffi_scan_null_handle_accessors() {
        assert_eq!(super::tgm_scan_count(ptr::null()), 0);
        let entry = super::tgm_scan_entry(ptr::null(), 0);
        assert_eq!(entry.offset, usize::MAX);
        assert_eq!(entry.length, 0);
    }

    #[test]
    fn ffi_scan_concatenated_messages() {
        let msg1 = ffi_encode_single_f32_tensor(&[1.0f32], "");
        let msg2 = ffi_encode_single_f32_tensor(&[2.0f32], "");
        let mut concat = msg1.clone();
        concat.extend_from_slice(&msg2);

        let mut result: *mut super::TgmScanResult = ptr::null_mut();
        let err = super::tgm_scan(concat.as_ptr(), concat.len(), &mut result);
        assert!(matches!(err, super::TgmError::Ok));

        assert_eq!(super::tgm_scan_count(result), 2);

        let e0 = super::tgm_scan_entry(result, 0);
        assert_eq!(e0.offset, 0);
        assert_eq!(e0.length, msg1.len());

        let e1 = super::tgm_scan_entry(result, 1);
        assert_eq!(e1.offset, msg1.len());
        assert_eq!(e1.length, msg2.len());

        super::tgm_scan_free(result);
    }

    // ── tgm_file_* functions ──

    #[test]
    fn ffi_file_create_append_count_decode_close() {
        let dir = std::env::temp_dir();
        let path = dir.join("ffi_test_file.tgm");
        let _ = std::fs::remove_file(&path);

        let c_path = CString::new(path.to_str().unwrap()).unwrap();

        // Create
        let mut file: *mut super::TgmFile = ptr::null_mut();
        let err = super::tgm_file_create(c_path.as_ptr(), &mut file);
        assert!(matches!(err, super::TgmError::Ok));
        assert!(!file.is_null());

        // Append a message
        let values = [10.0f32, 20.0, 30.0];
        let shape_str = format!("[{}]", values.len());
        let json = format!(
            r#"{{"version":3,"descriptors":[{{"type":"ntensor","ndim":1,"shape":{shape},"strides":[1],"dtype":"float32","byte_order":"{bo}","encoding":"none","filter":"none","compression":"none"}}]}}"#,
            shape = shape_str,
            bo = if cfg!(target_endian = "little") {
                "little"
            } else {
                "big"
            },
        );
        let c_json = CString::new(json).unwrap();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let data_ptr: *const u8 = data.as_ptr();
        let data_len: usize = data.len();

        let err = super::tgm_file_append(
            file,
            c_json.as_ptr(),
            &data_ptr as *const *const u8,
            &data_len as *const usize,
            1,
            ptr::null(),
            0,
        );
        assert!(matches!(err, super::TgmError::Ok));

        // Check path accessor
        let path_ptr = super::tgm_file_path(file);
        assert!(!path_ptr.is_null());
        let path_str = unsafe { CStr::from_ptr(path_ptr) }.to_str().unwrap();
        assert!(path_str.contains("ffi_test_file.tgm"));

        super::tgm_file_close(file);

        // Re-open for reading
        let mut file2: *mut super::TgmFile = ptr::null_mut();
        let err = super::tgm_file_open(c_path.as_ptr(), &mut file2);
        assert!(matches!(err, super::TgmError::Ok));

        // Message count
        let mut count: usize = 0;
        let err = super::tgm_file_message_count(file2, &mut count);
        assert!(matches!(err, super::TgmError::Ok));
        assert_eq!(count, 1);

        // Decode message
        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_file_decode_message(file2, 0, 0, 0, 0, &mut msg);
        assert!(matches!(err, super::TgmError::Ok));

        assert_eq!(super::tgm_message_num_objects(msg), 1);
        let mut data_len2: usize = 0;
        let dp = super::tgm_object_data(msg, 0, &mut data_len2);
        assert!(!dp.is_null());
        let decoded_bytes = unsafe { slice::from_raw_parts(dp, data_len2) };
        let decoded_values: Vec<f32> = decoded_bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(decoded_values, values);

        super::tgm_message_free(msg);

        // Read raw message
        let mut raw = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_file_read_message(file2, 0, &mut raw);
        assert!(matches!(err, super::TgmError::Ok));
        assert!(!raw.data.is_null());
        assert!(raw.len > 0);
        super::tgm_bytes_free(raw);

        super::tgm_file_close(file2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ffi_file_open_nonexistent() {
        let c_path = CString::new("/nonexistent/file.tgm").unwrap();
        let mut file: *mut super::TgmFile = ptr::null_mut();
        let err = super::tgm_file_open(c_path.as_ptr(), &mut file);
        assert!(!matches!(err, super::TgmError::Ok));
    }

    #[test]
    fn ffi_file_null_args() {
        let mut file: *mut super::TgmFile = ptr::null_mut();

        // open null path
        let err = super::tgm_file_open(ptr::null(), &mut file);
        assert!(matches!(err, super::TgmError::InvalidArg));

        // open null out
        let c_path = CString::new("/tmp/test.tgm").unwrap();
        let err = super::tgm_file_open(c_path.as_ptr(), ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));

        // create null path
        let err = super::tgm_file_create(ptr::null(), &mut file);
        assert!(matches!(err, super::TgmError::InvalidArg));

        // create null out
        let err = super::tgm_file_create(c_path.as_ptr(), ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));

        // message_count null args
        let err = super::tgm_file_message_count(ptr::null_mut(), ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));

        // decode_message null args
        let err = super::tgm_file_decode_message(ptr::null_mut(), 0, 0, 0, 0, ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));

        // read_message null args
        let err = super::tgm_file_read_message(ptr::null_mut(), 0, ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));

        // append null args
        let err = super::tgm_file_append(
            ptr::null_mut(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null(),
            0,
        );
        assert!(matches!(err, super::TgmError::InvalidArg));

        // append_raw null args
        let err = super::tgm_file_append_raw(ptr::null_mut(), ptr::null(), 0);
        assert!(matches!(err, super::TgmError::InvalidArg));

        // path null
        assert!(super::tgm_file_path(ptr::null()).is_null());
    }

    #[test]
    fn ffi_file_append_raw_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("ffi_test_append_raw.tgm");
        let _ = std::fs::remove_file(&path);

        let c_path = CString::new(path.to_str().unwrap()).unwrap();

        // Create
        let mut file: *mut super::TgmFile = ptr::null_mut();
        let err = super::tgm_file_create(c_path.as_ptr(), &mut file);
        assert!(matches!(err, super::TgmError::Ok));

        // Encode a message in memory, then append raw bytes
        let encoded = ffi_encode_single_f32_tensor(&[1.0f32, 2.0], "");
        let err = super::tgm_file_append_raw(file, encoded.as_ptr(), encoded.len());
        assert!(matches!(err, super::TgmError::Ok));

        // Count
        let mut count: usize = 0;
        let err = super::tgm_file_message_count(file, &mut count);
        assert!(matches!(err, super::TgmError::Ok));
        assert_eq!(count, 1);

        super::tgm_file_close(file);
        let _ = std::fs::remove_file(&path);
    }

    // ── tgm_streaming_encoder_* ──

    #[test]
    fn ffi_streaming_encoder_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("ffi_streaming_test.tgm");
        let _ = std::fs::remove_file(&path);

        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let meta_json = CString::new(r#"{"version":3}"#).unwrap();

        // Create
        let mut enc: *mut super::TgmStreamingEncoder = ptr::null_mut();
        let err = super::tgm_streaming_encoder_create(
            c_path.as_ptr(),
            meta_json.as_ptr(),
            ptr::null(), // no hash
            0,           // threads
            &mut enc,
        );
        assert!(matches!(err, super::TgmError::Ok));
        assert!(!enc.is_null());

        // Write an object
        let values = [100.0f32, 200.0, 300.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let desc_json = CString::new(format!(
            r#"{{"type":"ntensor","ndim":1,"shape":[{len}],"strides":[1],"dtype":"float32","byte_order":"{bo}","encoding":"none","filter":"none","compression":"none"}}"#,
            len = values.len(),
            bo = if cfg!(target_endian = "little") { "little" } else { "big" },
        )).unwrap();

        let err =
            super::tgm_streaming_encoder_write(enc, desc_json.as_ptr(), data.as_ptr(), data.len());
        assert!(matches!(err, super::TgmError::Ok));

        // Count
        assert_eq!(super::tgm_streaming_encoder_count(enc), 1);

        // Finish
        let err = super::tgm_streaming_encoder_finish(enc);
        assert!(matches!(err, super::TgmError::Ok));

        // Double finish should fail
        let err = super::tgm_streaming_encoder_finish(enc);
        assert!(matches!(err, super::TgmError::InvalidArg));

        // Count after finish
        assert_eq!(super::tgm_streaming_encoder_count(enc), 0);

        // Free
        super::tgm_streaming_encoder_free(enc);

        // Read back and verify
        let mut file: *mut super::TgmFile = ptr::null_mut();
        let err = super::tgm_file_open(c_path.as_ptr(), &mut file);
        assert!(matches!(err, super::TgmError::Ok));

        let mut count: usize = 0;
        let err = super::tgm_file_message_count(file, &mut count);
        assert!(matches!(err, super::TgmError::Ok));
        assert_eq!(count, 1);

        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_file_decode_message(file, 0, 0, 0, 0, &mut msg);
        assert!(matches!(err, super::TgmError::Ok));

        let mut data_len: usize = 0;
        let dp = super::tgm_object_data(msg, 0, &mut data_len);
        let decoded_bytes = unsafe { slice::from_raw_parts(dp, data_len) };
        let decoded_values: Vec<f32> = decoded_bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(decoded_values, values);

        super::tgm_message_free(msg);
        super::tgm_file_close(file);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ffi_streaming_encoder_null_args() {
        let mut enc: *mut super::TgmStreamingEncoder = ptr::null_mut();

        // create with null path
        let meta = CString::new(r#"{"version":3}"#).unwrap();
        let err = super::tgm_streaming_encoder_create(
            ptr::null(),
            meta.as_ptr(),
            ptr::null(),
            0,
            &mut enc,
        );
        assert!(matches!(err, super::TgmError::InvalidArg));

        // create with null metadata
        let p = CString::new("/tmp/dummy.tgm").unwrap();
        let err =
            super::tgm_streaming_encoder_create(p.as_ptr(), ptr::null(), ptr::null(), 0, &mut enc);
        assert!(matches!(err, super::TgmError::InvalidArg));

        // create with null out
        let err = super::tgm_streaming_encoder_create(
            p.as_ptr(),
            meta.as_ptr(),
            ptr::null(),
            0,
            ptr::null_mut(),
        );
        assert!(matches!(err, super::TgmError::InvalidArg));

        // write null enc
        let desc = CString::new(r#"{}"#).unwrap();
        let data = [0u8; 4];
        let err = super::tgm_streaming_encoder_write(
            ptr::null_mut(),
            desc.as_ptr(),
            data.as_ptr(),
            data.len(),
        );
        assert!(matches!(err, super::TgmError::InvalidArg));

        // write null descriptor
        // Need a valid encoder for this — skip as it requires file creation

        // finish null
        let err = super::tgm_streaming_encoder_finish(ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));

        // count null
        assert_eq!(super::tgm_streaming_encoder_count(ptr::null()), 0);

        // free null — should not crash
        super::tgm_streaming_encoder_free(ptr::null_mut());
    }

    #[test]
    fn ffi_streaming_encoder_write_null_data() {
        let dir = std::env::temp_dir();
        let path = dir.join("ffi_streaming_null_data.tgm");
        let _ = std::fs::remove_file(&path);

        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let meta_json = CString::new(r#"{"version":3}"#).unwrap();

        let mut enc: *mut super::TgmStreamingEncoder = ptr::null_mut();
        let err = super::tgm_streaming_encoder_create(
            c_path.as_ptr(),
            meta_json.as_ptr(),
            ptr::null(),
            0,
            &mut enc,
        );
        assert!(matches!(err, super::TgmError::Ok));

        let desc = CString::new(r#"{"type":"ntensor","ndim":1,"shape":[1],"strides":[1],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"none"}"#).unwrap();
        let err = super::tgm_streaming_encoder_write(enc, desc.as_ptr(), ptr::null(), 4);
        assert!(matches!(err, super::TgmError::InvalidArg));

        // Write with null descriptor json
        let data = [0u8; 4];
        let err = super::tgm_streaming_encoder_write(enc, ptr::null(), data.as_ptr(), data.len());
        assert!(matches!(err, super::TgmError::InvalidArg));

        super::tgm_streaming_encoder_free(enc);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ffi_streaming_encoder_with_preceder() {
        let dir = std::env::temp_dir();
        let path = dir.join("ffi_streaming_preceder.tgm");
        let _ = std::fs::remove_file(&path);

        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let meta_json = CString::new(r#"{"version":3}"#).unwrap();

        let mut enc: *mut super::TgmStreamingEncoder = ptr::null_mut();
        let err = super::tgm_streaming_encoder_create(
            c_path.as_ptr(),
            meta_json.as_ptr(),
            ptr::null(),
            0,
            &mut enc,
        );
        assert!(matches!(err, super::TgmError::Ok));

        // Write preceder
        let preceder_json = CString::new(r#"{"param":"2t","source":"test"}"#).unwrap();
        let err = super::tgm_streaming_encoder_write_preceder(enc, preceder_json.as_ptr());
        assert!(matches!(err, super::TgmError::Ok));

        // Write object after preceder
        let values = [42.0f32];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let desc_json = CString::new(format!(
            r#"{{"type":"ntensor","ndim":1,"shape":[1],"strides":[1],"dtype":"float32","byte_order":"{bo}","encoding":"none","filter":"none","compression":"none"}}"#,
            bo = if cfg!(target_endian = "little") { "little" } else { "big" },
        )).unwrap();

        let err =
            super::tgm_streaming_encoder_write(enc, desc_json.as_ptr(), data.as_ptr(), data.len());
        assert!(matches!(err, super::TgmError::Ok));

        let err = super::tgm_streaming_encoder_finish(enc);
        assert!(matches!(err, super::TgmError::Ok));
        super::tgm_streaming_encoder_free(enc);

        // Re-open and verify the metadata contains the preceder keys
        let mut file: *mut super::TgmFile = ptr::null_mut();
        let err = super::tgm_file_open(c_path.as_ptr(), &mut file);
        assert!(matches!(err, super::TgmError::Ok));

        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_file_decode_message(file, 0, 0, 0, 0, &mut msg);
        assert!(matches!(err, super::TgmError::Ok));

        let mut meta: *mut super::TgmMetadata = ptr::null_mut();
        let err = super::tgm_message_metadata(msg, &mut meta);
        assert!(matches!(err, super::TgmError::Ok));

        let key = CString::new("param").unwrap();
        let val_ptr = super::tgm_metadata_get_string(meta, key.as_ptr());
        assert!(!val_ptr.is_null());
        let val_str = unsafe { CStr::from_ptr(val_ptr) }.to_str().unwrap();
        assert_eq!(val_str, "2t");

        super::tgm_metadata_free(meta);
        super::tgm_message_free(msg);
        super::tgm_file_close(file);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ffi_streaming_encoder_write_preceder_null_args() {
        let err = super::tgm_streaming_encoder_write_preceder(ptr::null_mut(), ptr::null());
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn ffi_streaming_encoder_write_pre_encoded_null_args() {
        let err = super::tgm_streaming_encoder_write_pre_encoded(
            ptr::null_mut(),
            ptr::null(),
            ptr::null(),
            0,
        );
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    // ── tgm_compute_hash ──

    #[test]
    fn ffi_compute_hash_xxh3() {
        let data = b"hello world";
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_compute_hash(
            data.as_ptr(),
            data.len(),
            ptr::null(), // default = xxh3
            &mut out,
        );
        assert!(matches!(err, super::TgmError::Ok));
        assert!(!out.data.is_null());
        assert!(out.len > 0);

        let hex = unsafe { std::str::from_utf8(slice::from_raw_parts(out.data, out.len)).unwrap() };
        // xxh3 produces a hex string
        assert!(!hex.is_empty());
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));

        super::tgm_bytes_free(out);
    }

    #[test]
    fn ffi_compute_hash_explicit_xxh3() {
        let data = b"test data";
        let algo = CString::new("xxh3").unwrap();
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_compute_hash(data.as_ptr(), data.len(), algo.as_ptr(), &mut out);
        assert!(matches!(err, super::TgmError::Ok));
        assert!(out.len > 0);
        super::tgm_bytes_free(out);
    }

    #[test]
    fn ffi_compute_hash_null_data() {
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_compute_hash(ptr::null(), 0, ptr::null(), &mut out);
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn ffi_compute_hash_null_out() {
        let data = b"hello";
        let err = super::tgm_compute_hash(data.as_ptr(), data.len(), ptr::null(), ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn ffi_compute_hash_invalid_algo() {
        let data = b"hello";
        let algo = CString::new("bogus_algo").unwrap();
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_compute_hash(data.as_ptr(), data.len(), algo.as_ptr(), &mut out);
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    // ── tgm_simple_packing_compute_params ──

    #[test]
    fn ffi_simple_packing_compute_params() {
        let values = [100.0f64, 200.0, 300.0, 400.0];
        let mut ref_val: f64 = 0.0;
        let mut bin_scale: i32 = 0;
        let err = super::tgm_simple_packing_compute_params(
            values.as_ptr(),
            values.len(),
            16,
            0,
            &mut ref_val,
            &mut bin_scale,
        );
        assert!(matches!(err, super::TgmError::Ok));
        // Reference value should be <= min(values)
        assert!(ref_val <= 100.0);
    }

    #[test]
    fn ffi_simple_packing_compute_params_null_args() {
        let mut ref_val: f64 = 0.0;
        let mut bin_scale: i32 = 0;

        // null values
        let err = super::tgm_simple_packing_compute_params(
            ptr::null(),
            0,
            16,
            0,
            &mut ref_val,
            &mut bin_scale,
        );
        assert!(matches!(err, super::TgmError::InvalidArg));

        // null out_reference_value
        let values = [1.0f64];
        let err = super::tgm_simple_packing_compute_params(
            values.as_ptr(),
            values.len(),
            16,
            0,
            ptr::null_mut(),
            &mut bin_scale,
        );
        assert!(matches!(err, super::TgmError::InvalidArg));

        // null out_binary_scale_factor
        let err = super::tgm_simple_packing_compute_params(
            values.as_ptr(),
            values.len(),
            16,
            0,
            &mut ref_val,
            ptr::null_mut(),
        );
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    // ── tgm_last_error ──

    #[test]
    fn ffi_last_error_after_success() {
        // After a successful encode, last_error should remain from whatever was
        // set before (or NULL). We don't clear on success.
        let values = [1.0f32];
        let _ = ffi_encode_single_f32_tensor(&values, "");
        // No crash, test passes
    }

    #[test]
    fn ffi_last_error_after_failure() {
        // Trigger an error
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let _ = super::tgm_encode(
            ptr::null(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            &mut out,
        );

        let err_ptr = super::tgm_last_error();
        assert!(!err_ptr.is_null());
        let err_str = unsafe { CStr::from_ptr(err_ptr) }.to_str().unwrap();
        assert!(err_str.contains("null"));
    }

    // ── tgm_error_string ──

    #[test]
    fn ffi_error_string_all_variants() {
        let check = |err: super::TgmError, expected: &str| {
            let ptr = super::tgm_error_string(err);
            assert!(!ptr.is_null());
            let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
            assert_eq!(s, expected);
        };

        check(super::TgmError::Ok, "ok");
        check(super::TgmError::Framing, "framing error");
        check(super::TgmError::Metadata, "metadata error");
        check(super::TgmError::Encoding, "encoding error");
        check(super::TgmError::Compression, "compression error");
        check(super::TgmError::Object, "object error");
        check(super::TgmError::Io, "I/O error");
        check(super::TgmError::HashMismatch, "hash mismatch");
        check(super::TgmError::InvalidArg, "invalid argument");
        check(super::TgmError::EndOfIter, "end of iteration");
        check(super::TgmError::Remote, "remote error");
    }

    // ── tgm_bytes_free safety ──

    #[test]
    fn ffi_bytes_free_null_data() {
        let buf = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        super::tgm_bytes_free(buf); // should not crash
    }

    // ── tgm_message_free / tgm_metadata_free / tgm_scan_free null safety ──

    #[test]
    fn ffi_free_null_handles() {
        super::tgm_message_free(ptr::null_mut());
        super::tgm_metadata_free(ptr::null_mut());
        super::tgm_scan_free(ptr::null_mut());
        super::tgm_file_close(ptr::null_mut());
        super::tgm_streaming_encoder_free(ptr::null_mut());
        // Should all be no-ops, no crash
    }

    // ── tgm_encode_pre_encoded ──

    #[test]
    fn ffi_encode_pre_encoded_null_args() {
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_encode_pre_encoded(
            ptr::null(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            &mut out,
        );
        assert!(matches!(err, super::TgmError::InvalidArg));

        let json = CString::new(r#"{"version":3,"descriptors":[]}"#).unwrap();
        let err = super::tgm_encode_pre_encoded(
            json.as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            ptr::null_mut(),
        );
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn ffi_encode_pre_encoded_round_trip() {
        // Encode with pre_encoded: for "none" encoding, the raw bytes are the payload
        let values = [5.0f32, 6.0, 7.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

        let json = format!(
            r#"{{"version":3,"descriptors":[{{"type":"ntensor","ndim":1,"shape":[{len}],"strides":[1],"dtype":"float32","byte_order":"{bo}","encoding":"none","filter":"none","compression":"none"}}]}}"#,
            len = values.len(),
            bo = if cfg!(target_endian = "little") {
                "little"
            } else {
                "big"
            },
        );
        let c_json = CString::new(json).unwrap();
        let data_ptr: *const u8 = data.as_ptr();
        let data_len: usize = data.len();

        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_encode_pre_encoded(
            c_json.as_ptr(),
            &data_ptr as *const *const u8,
            &data_len as *const usize,
            1,
            ptr::null(),
            0,
            &mut out,
        );
        assert!(matches!(err, super::TgmError::Ok));

        let encoded = unsafe { slice::from_raw_parts(out.data, out.len) }.to_vec();
        super::tgm_bytes_free(out);

        // Decode
        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_decode(encoded.as_ptr(), encoded.len(), 0, 0, 0, &mut msg);
        assert!(matches!(err, super::TgmError::Ok));

        let mut dl: usize = 0;
        let dp = super::tgm_object_data(msg, 0, &mut dl);
        let decoded_bytes = unsafe { slice::from_raw_parts(dp, dl) };
        let decoded_values: Vec<f32> = decoded_bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(decoded_values, values);

        super::tgm_message_free(msg);
    }

    // ── tgm_buffer_iter_* ──

    #[test]
    fn ffi_buffer_iter_round_trip() {
        let msg1 = ffi_encode_single_f32_tensor(&[1.0f32], "");
        let msg2 = ffi_encode_single_f32_tensor(&[2.0f32], "");
        let mut concat = msg1.clone();
        concat.extend_from_slice(&msg2);

        let mut iter: *mut super::TgmBufferIter = ptr::null_mut();
        let err = super::tgm_buffer_iter_create(concat.as_ptr(), concat.len(), &mut iter);
        assert!(matches!(err, super::TgmError::Ok));
        assert!(!iter.is_null());

        assert_eq!(super::tgm_buffer_iter_count(iter), 2);

        // First message
        let mut out_buf: *const u8 = ptr::null();
        let mut out_len: usize = 0;
        let err = super::tgm_buffer_iter_next(iter, &mut out_buf, &mut out_len);
        assert!(matches!(err, super::TgmError::Ok));
        assert!(!out_buf.is_null());
        assert_eq!(out_len, msg1.len());

        // Second message
        let err = super::tgm_buffer_iter_next(iter, &mut out_buf, &mut out_len);
        assert!(matches!(err, super::TgmError::Ok));
        assert_eq!(out_len, msg2.len());

        // End of iteration
        let err = super::tgm_buffer_iter_next(iter, &mut out_buf, &mut out_len);
        assert!(matches!(err, super::TgmError::EndOfIter));

        super::tgm_buffer_iter_free(iter);
    }

    #[test]
    fn ffi_buffer_iter_null_args() {
        let mut iter: *mut super::TgmBufferIter = ptr::null_mut();
        let err = super::tgm_buffer_iter_create(ptr::null(), 0, &mut iter);
        assert!(matches!(err, super::TgmError::InvalidArg));

        let data = [0u8; 10];
        let err = super::tgm_buffer_iter_create(data.as_ptr(), data.len(), ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));

        // next with null iter
        let mut out_buf: *const u8 = ptr::null();
        let mut out_len: usize = 0;
        let err = super::tgm_buffer_iter_next(ptr::null_mut(), &mut out_buf, &mut out_len);
        assert!(matches!(err, super::TgmError::InvalidArg));

        // next with null out_buf
        // We need a valid iter for this, but let's test the easy null cases
        let err = super::tgm_buffer_iter_next(ptr::null_mut(), ptr::null_mut(), &mut out_len);
        assert!(matches!(err, super::TgmError::InvalidArg));

        // count null
        assert_eq!(super::tgm_buffer_iter_count(ptr::null()), 0);

        // free null — no crash
        super::tgm_buffer_iter_free(ptr::null_mut());
    }

    // ── tgm_object_iter_* ──

    #[test]
    fn ffi_object_iter_round_trip() {
        let encoded = ffi_encode_single_f32_tensor(&[10.0f32, 20.0], "");

        let mut iter: *mut super::TgmObjectIter = ptr::null_mut();
        let err = super::tgm_object_iter_create(
            encoded.as_ptr(),
            encoded.len(),
            0, // no hash verify
            0, // no native byte order
            &mut iter,
        );
        assert!(matches!(err, super::TgmError::Ok));
        assert!(!iter.is_null());

        // Get the single object
        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_object_iter_next(iter, &mut msg);
        assert!(matches!(err, super::TgmError::Ok));
        assert!(!msg.is_null());

        assert_eq!(super::tgm_message_num_objects(msg), 1);
        assert_eq!(super::tgm_message_version(msg), 3);

        let mut data_len: usize = 0;
        let dp = super::tgm_object_data(msg, 0, &mut data_len);
        assert!(!dp.is_null());
        assert_eq!(data_len, 8); // 2 × f32

        super::tgm_message_free(msg);

        // Iteration should be exhausted
        let mut msg2: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_object_iter_next(iter, &mut msg2);
        assert!(matches!(err, super::TgmError::EndOfIter));

        super::tgm_object_iter_free(iter);
    }

    #[test]
    fn ffi_object_iter_null_args() {
        let mut iter: *mut super::TgmObjectIter = ptr::null_mut();
        let err = super::tgm_object_iter_create(ptr::null(), 0, 0, 0, &mut iter);
        assert!(matches!(err, super::TgmError::InvalidArg));

        let data = [0u8; 10];
        let err = super::tgm_object_iter_create(data.as_ptr(), data.len(), 0, 0, ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));

        // next null iter
        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_object_iter_next(ptr::null_mut(), &mut msg);
        assert!(matches!(err, super::TgmError::InvalidArg));

        // next null out
        let err = super::tgm_object_iter_next(ptr::null_mut(), ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));

        // free null — no crash
        super::tgm_object_iter_free(ptr::null_mut());
    }

    // ── tgm_file_iter_* ──

    #[test]
    fn ffi_file_iter_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("ffi_file_iter_test.tgm");
        let _ = std::fs::remove_file(&path);

        let c_path = CString::new(path.to_str().unwrap()).unwrap();

        // Create file with 2 messages
        let mut file: *mut super::TgmFile = ptr::null_mut();
        let err = super::tgm_file_create(c_path.as_ptr(), &mut file);
        assert!(matches!(err, super::TgmError::Ok));

        let msg1 = ffi_encode_single_f32_tensor(&[1.0f32], "");
        let msg2 = ffi_encode_single_f32_tensor(&[2.0f32], "");
        let err = super::tgm_file_append_raw(file, msg1.as_ptr(), msg1.len());
        assert!(matches!(err, super::TgmError::Ok));
        let err = super::tgm_file_append_raw(file, msg2.as_ptr(), msg2.len());
        assert!(matches!(err, super::TgmError::Ok));

        // Create iterator
        let mut iter: *mut super::TgmFileIter = ptr::null_mut();
        let err = super::tgm_file_iter_create(file, &mut iter);
        assert!(matches!(err, super::TgmError::Ok));
        assert!(!iter.is_null());

        // First message
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_file_iter_next(iter, &mut out);
        assert!(matches!(err, super::TgmError::Ok));
        assert!(!out.data.is_null());
        assert_eq!(out.len, msg1.len());
        super::tgm_bytes_free(out);

        // Second message
        let mut out2 = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_file_iter_next(iter, &mut out2);
        assert!(matches!(err, super::TgmError::Ok));
        super::tgm_bytes_free(out2);

        // End
        let mut out3 = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_file_iter_next(iter, &mut out3);
        assert!(matches!(err, super::TgmError::EndOfIter));

        super::tgm_file_iter_free(iter);
        super::tgm_file_close(file);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ffi_file_iter_null_args() {
        let mut iter: *mut super::TgmFileIter = ptr::null_mut();
        let err = super::tgm_file_iter_create(ptr::null_mut(), &mut iter);
        assert!(matches!(err, super::TgmError::InvalidArg));

        let err = super::tgm_file_iter_create(ptr::null_mut(), ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));

        // next null
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_file_iter_next(ptr::null_mut(), &mut out);
        assert!(matches!(err, super::TgmError::InvalidArg));

        let err = super::tgm_file_iter_next(ptr::null_mut(), ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));

        // free null — no crash
        super::tgm_file_iter_free(ptr::null_mut());
    }

    // ── tgm_encode zero objects (metadata-only message) ──

    #[test]
    fn ffi_encode_decode_zero_objects() {
        let json = CString::new(r#"{"version":3,"descriptors":[],"source":"empty"}"#).unwrap();
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_encode(
            json.as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            &mut out,
        );
        assert!(matches!(err, super::TgmError::Ok));

        let encoded = unsafe { slice::from_raw_parts(out.data, out.len) }.to_vec();
        super::tgm_bytes_free(out);

        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_decode(encoded.as_ptr(), encoded.len(), 0, 0, 0, &mut msg);
        assert!(matches!(err, super::TgmError::Ok));

        assert_eq!(super::tgm_message_version(msg), 3);
        assert_eq!(super::tgm_message_num_objects(msg), 0);

        super::tgm_message_free(msg);
    }

    // ── tgm_streaming_encoder with hash ──

    #[test]
    #[ignore = "v3: hash moved to frame footer — re-enable in phase 6"]
    fn ffi_streaming_encoder_with_hash() {
        let dir = std::env::temp_dir();
        let path = dir.join("ffi_streaming_hash.tgm");
        let _ = std::fs::remove_file(&path);

        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let meta_json = CString::new(r#"{"version":3}"#).unwrap();
        let hash_algo = CString::new("xxh3").unwrap();

        let mut enc: *mut super::TgmStreamingEncoder = ptr::null_mut();
        let err = super::tgm_streaming_encoder_create(
            c_path.as_ptr(),
            meta_json.as_ptr(),
            hash_algo.as_ptr(),
            0,
            &mut enc,
        );
        assert!(matches!(err, super::TgmError::Ok));

        let values = [1.0f32, 2.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let desc_json = CString::new(format!(
            r#"{{"type":"ntensor","ndim":1,"shape":[{len}],"strides":[1],"dtype":"float32","byte_order":"{bo}","encoding":"none","filter":"none","compression":"none"}}"#,
            len = values.len(),
            bo = if cfg!(target_endian = "little") { "little" } else { "big" },
        )).unwrap();

        let err =
            super::tgm_streaming_encoder_write(enc, desc_json.as_ptr(), data.as_ptr(), data.len());
        assert!(matches!(err, super::TgmError::Ok));

        let err = super::tgm_streaming_encoder_finish(enc);
        assert!(matches!(err, super::TgmError::Ok));
        super::tgm_streaming_encoder_free(enc);

        // Read back and verify hash
        let mut file: *mut super::TgmFile = ptr::null_mut();
        let err = super::tgm_file_open(c_path.as_ptr(), &mut file);
        assert!(matches!(err, super::TgmError::Ok));

        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_file_decode_message(file, 0, 1, 0, 0, &mut msg);
        assert!(matches!(err, super::TgmError::Ok));

        assert_eq!(super::tgm_payload_has_hash(msg, 0), 1);

        super::tgm_message_free(msg);
        super::tgm_file_close(file);
        let _ = std::fs::remove_file(&path);
    }

    // ── tgm_streaming_encoder_create with invalid hash algo ──

    #[test]
    fn ffi_streaming_encoder_invalid_hash_algo() {
        let dir = std::env::temp_dir();
        let path = dir.join("ffi_streaming_bad_hash.tgm");
        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let meta_json = CString::new(r#"{"version":3}"#).unwrap();
        let bad_algo = CString::new("bogus_hash").unwrap();

        let mut enc: *mut super::TgmStreamingEncoder = ptr::null_mut();
        let err = super::tgm_streaming_encoder_create(
            c_path.as_ptr(),
            meta_json.as_ptr(),
            bad_algo.as_ptr(),
            0,
            &mut enc,
        );
        assert!(matches!(err, super::TgmError::InvalidArg));
        let _ = std::fs::remove_file(&path);
    }

    // ── tgm_streaming_encoder_create with invalid metadata JSON ──

    #[test]
    fn ffi_streaming_encoder_invalid_metadata() {
        let dir = std::env::temp_dir();
        let path = dir.join("ffi_streaming_bad_meta.tgm");
        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let bad_meta = CString::new("not json").unwrap();

        let mut enc: *mut super::TgmStreamingEncoder = ptr::null_mut();
        let err = super::tgm_streaming_encoder_create(
            c_path.as_ptr(),
            bad_meta.as_ptr(),
            ptr::null(),
            0,
            &mut enc,
        );
        assert!(matches!(err, super::TgmError::Metadata));
        let _ = std::fs::remove_file(&path);
    }

    // ── Multiple objects encode/decode ──

    #[test]
    fn ffi_encode_decode_multiple_objects() {
        let vals1 = [1.0f32, 2.0];
        let vals2 = [10.0f32, 20.0, 30.0];
        let bo = if cfg!(target_endian = "little") {
            "little"
        } else {
            "big"
        };

        let json = format!(
            r#"{{"version":3,"descriptors":[{{"type":"ntensor","ndim":1,"shape":[{len1}],"strides":[1],"dtype":"float32","byte_order":"{bo}","encoding":"none","filter":"none","compression":"none"}},{{"type":"ntensor","ndim":1,"shape":[{len2}],"strides":[1],"dtype":"float32","byte_order":"{bo}","encoding":"none","filter":"none","compression":"none"}}]}}"#,
            len1 = vals1.len(),
            len2 = vals2.len(),
            bo = bo,
        );

        let data1: Vec<u8> = vals1.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let data2: Vec<u8> = vals2.iter().flat_map(|v| v.to_ne_bytes()).collect();

        let c_json = CString::new(json).unwrap();
        let data_ptrs: [*const u8; 2] = [data1.as_ptr(), data2.as_ptr()];
        let data_lens: [usize; 2] = [data1.len(), data2.len()];

        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_encode(
            c_json.as_ptr(),
            data_ptrs.as_ptr(),
            data_lens.as_ptr(),
            2,
            ptr::null(),
            0,
            &mut out,
        );
        assert!(matches!(err, super::TgmError::Ok));

        let encoded = unsafe { slice::from_raw_parts(out.data, out.len) }.to_vec();
        super::tgm_bytes_free(out);

        // Decode all
        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_decode(encoded.as_ptr(), encoded.len(), 0, 0, 0, &mut msg);
        assert!(matches!(err, super::TgmError::Ok));
        assert_eq!(super::tgm_message_num_objects(msg), 2);

        // Object 0
        let mut dl: usize = 0;
        let dp = super::tgm_object_data(msg, 0, &mut dl);
        let decoded0: Vec<f32> = unsafe { slice::from_raw_parts(dp, dl) }
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(decoded0, vals1);

        // Object 1
        let dp1 = super::tgm_object_data(msg, 1, &mut dl);
        let decoded1: Vec<f32> = unsafe { slice::from_raw_parts(dp1, dl) }
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(decoded1, vals2);

        // Shape check
        let shape0 = super::tgm_object_shape(msg, 0);
        assert_eq!(unsafe { *shape0 }, vals1.len() as u64);
        let shape1 = super::tgm_object_shape(msg, 1);
        assert_eq!(unsafe { *shape1 }, vals2.len() as u64);

        super::tgm_message_free(msg);
    }

    // ── tgm_encode with base metadata ──

    #[test]
    fn ffi_encode_decode_with_base_metadata() {
        let bo = if cfg!(target_endian = "little") {
            "little"
        } else {
            "big"
        };
        let json = format!(
            r#"{{"version":3,"base":[{{"param":"2t","level":"surface"}}],"descriptors":[{{"type":"ntensor","ndim":1,"shape":[2],"strides":[1],"dtype":"float32","byte_order":"{bo}","encoding":"none","filter":"none","compression":"none"}}]}}"#,
            bo = bo,
        );

        let data: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_ne_bytes()).collect();
        let c_json = CString::new(json).unwrap();
        let data_ptr: *const u8 = data.as_ptr();
        let data_len: usize = data.len();

        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_encode(
            c_json.as_ptr(),
            &data_ptr as *const *const u8,
            &data_len as *const usize,
            1,
            ptr::null(),
            0,
            &mut out,
        );
        assert!(matches!(err, super::TgmError::Ok));

        let encoded = unsafe { slice::from_raw_parts(out.data, out.len) }.to_vec();
        super::tgm_bytes_free(out);

        // Decode metadata and check base keys
        let mut meta: *mut super::TgmMetadata = ptr::null_mut();
        let err = super::tgm_decode_metadata(encoded.as_ptr(), encoded.len(), &mut meta);
        assert!(matches!(err, super::TgmError::Ok));

        let key = CString::new("param").unwrap();
        let val_ptr = super::tgm_metadata_get_string(meta, key.as_ptr());
        assert!(!val_ptr.is_null());
        let val_str = unsafe { CStr::from_ptr(val_ptr) }.to_str().unwrap();
        assert_eq!(val_str, "2t");

        let key2 = CString::new("level").unwrap();
        let val_ptr2 = super::tgm_metadata_get_string(meta, key2.as_ptr());
        assert!(!val_ptr2.is_null());
        let val_str2 = unsafe { CStr::from_ptr(val_ptr2) }.to_str().unwrap();
        assert_eq!(val_str2, "surface");

        super::tgm_metadata_free(meta);
    }

    // ── tgm_compute_hash deterministic ──

    #[test]
    fn ffi_compute_hash_deterministic() {
        let data = b"deterministic hash test";
        let mut out1 = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let mut out2 = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };

        let err = super::tgm_compute_hash(data.as_ptr(), data.len(), ptr::null(), &mut out1);
        assert!(matches!(err, super::TgmError::Ok));

        let err = super::tgm_compute_hash(data.as_ptr(), data.len(), ptr::null(), &mut out2);
        assert!(matches!(err, super::TgmError::Ok));

        let hex1 = unsafe { slice::from_raw_parts(out1.data, out1.len) };
        let hex2 = unsafe { slice::from_raw_parts(out2.data, out2.len) };
        assert_eq!(hex1, hex2);

        super::tgm_bytes_free(out1);
        super::tgm_bytes_free(out2);
    }

    // ── tgm_streaming_encoder_write_pre_encoded round-trip ──

    #[test]
    fn ffi_streaming_encoder_write_pre_encoded_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("ffi_streaming_pre_encoded.tgm");
        let _ = std::fs::remove_file(&path);

        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let meta_json = CString::new(r#"{"version":3}"#).unwrap();

        let mut enc: *mut super::TgmStreamingEncoder = ptr::null_mut();
        let err = super::tgm_streaming_encoder_create(
            c_path.as_ptr(),
            meta_json.as_ptr(),
            ptr::null(),
            0,
            &mut enc,
        );
        assert!(matches!(err, super::TgmError::Ok));

        let values = [7.0f32, 8.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let desc_json = CString::new(format!(
            r#"{{"type":"ntensor","ndim":1,"shape":[{len}],"strides":[1],"dtype":"float32","byte_order":"{bo}","encoding":"none","filter":"none","compression":"none"}}"#,
            len = values.len(),
            bo = if cfg!(target_endian = "little") { "little" } else { "big" },
        )).unwrap();

        let err = super::tgm_streaming_encoder_write_pre_encoded(
            enc,
            desc_json.as_ptr(),
            data.as_ptr(),
            data.len(),
        );
        assert!(matches!(err, super::TgmError::Ok));

        let err = super::tgm_streaming_encoder_finish(enc);
        assert!(matches!(err, super::TgmError::Ok));
        super::tgm_streaming_encoder_free(enc);

        // Read back and verify
        let mut file: *mut super::TgmFile = ptr::null_mut();
        let err = super::tgm_file_open(c_path.as_ptr(), &mut file);
        assert!(matches!(err, super::TgmError::Ok));

        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_file_decode_message(file, 0, 0, 0, 0, &mut msg);
        assert!(matches!(err, super::TgmError::Ok));

        let mut dl: usize = 0;
        let dp = super::tgm_object_data(msg, 0, &mut dl);
        let decoded: Vec<f32> = unsafe { slice::from_raw_parts(dp, dl) }
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(decoded, values);

        super::tgm_message_free(msg);
        super::tgm_file_close(file);
        let _ = std::fs::remove_file(&path);
    }

    // ── tgm_encode with invalid hash algo ──

    #[test]
    fn ffi_encode_invalid_hash_algo() {
        let json = CString::new(r#"{"version":3,"descriptors":[]}"#).unwrap();
        let bad_algo = CString::new("bogus").unwrap();
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };

        let err = super::tgm_encode(
            json.as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            bad_algo.as_ptr(),
            0,
            &mut out,
        );
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    // ── tgm_scan_entry OOB returns sentinel and sets error ──

    #[test]
    fn ffi_scan_entry_oob_returns_sentinel() {
        let encoded = ffi_encode_single_f32_tensor(&[1.0f32], "");

        let mut result: *mut super::TgmScanResult = ptr::null_mut();
        let err = super::tgm_scan(encoded.as_ptr(), encoded.len(), &mut result);
        assert!(matches!(err, super::TgmError::Ok));

        assert_eq!(super::tgm_scan_count(result), 1);

        // Valid index works
        let good = super::tgm_scan_entry(result, 0);
        assert_eq!(good.offset, 0);
        assert!(good.length > 0);

        // OOB index returns sentinel
        let bad = super::tgm_scan_entry(result, 1);
        assert_eq!(bad.offset, usize::MAX);
        assert_eq!(bad.length, 0);

        // Error message is set
        let err_ptr = super::tgm_last_error();
        assert!(!err_ptr.is_null());
        let err_str = unsafe { CStr::from_ptr(err_ptr) }.to_str().unwrap();
        assert!(
            err_str.contains("out of range"),
            "expected OOB error, got: {err_str}"
        );

        super::tgm_scan_free(result);
    }

    // ── collect_data_slices null ptr + len=0 safety ──

    #[test]
    fn ffi_encode_zero_length_null_data_accepted() {
        // Encode with a zero-element tensor where the data pointer could be null
        // but length is 0 — should succeed without UB.
        let json = CString::new(
            r#"{"version":3,"descriptors":[{"type":"ntensor","ndim":1,"shape":[0],"strides":[1],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"none"}]}"#,
        )
        .unwrap();
        let data_ptrs: [*const u8; 1] = [ptr::null()];
        let data_lens: [usize; 1] = [0];
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };

        let err = super::tgm_encode(
            json.as_ptr(),
            data_ptrs.as_ptr(),
            data_lens.as_ptr(),
            1,
            ptr::null(), // no hash
            0,           // threads
            &mut out,
        );
        assert!(
            matches!(err, super::TgmError::Ok),
            "encoding zero-length data with null pointer should succeed"
        );
        if !out.data.is_null() {
            super::tgm_bytes_free(out);
        }
    }

    // ═══ Coverage-closer tests ═════════════════════════════════════════

    // ── tgm_validate (previously zero tests) ───────────────────────────

    #[test]
    fn ffi_validate_null_out() {
        let err = super::tgm_validate(ptr::null(), 0, ptr::null(), 0, ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn ffi_validate_null_buf_with_nonzero_len() {
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_validate(ptr::null(), 42, ptr::null(), 0, &mut out);
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn ffi_validate_empty_buffer_ok() {
        // buf=null, len=0 → valid empty-buffer validation
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_validate(ptr::null(), 0, ptr::null(), 0, &mut out);
        assert!(matches!(err, super::TgmError::Ok));
        assert!(!out.data.is_null());
        assert!(out.len > 0);
        super::tgm_bytes_free(out);
    }

    #[test]
    fn ffi_validate_valid_message_all_levels() {
        let encoded = ffi_encode_single_f32_tensor(&[1.0f32, 2.0, 3.0, 4.0], "");
        for level_str in &["quick", "checksum", "default", "full"] {
            let level = CString::new(*level_str).unwrap();
            let mut out = super::TgmBytes {
                data: ptr::null_mut(),
                len: 0,
            };
            let err =
                super::tgm_validate(encoded.as_ptr(), encoded.len(), level.as_ptr(), 0, &mut out);
            assert!(matches!(err, super::TgmError::Ok), "level {level_str}");
            super::tgm_bytes_free(out);
        }
    }

    #[test]
    fn ffi_validate_canonical_flag() {
        let encoded = ffi_encode_single_f32_tensor(&[1.0f32], "");
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_validate(
            encoded.as_ptr(),
            encoded.len(),
            ptr::null(),
            1, // check_canonical
            &mut out,
        );
        assert!(matches!(err, super::TgmError::Ok));
        super::tgm_bytes_free(out);
    }

    #[test]
    fn ffi_validate_invalid_level_string() {
        let encoded = ffi_encode_single_f32_tensor(&[1.0f32], "");
        let bogus = CString::new("bogus-level-name").unwrap();
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_validate(encoded.as_ptr(), encoded.len(), bogus.as_ptr(), 0, &mut out);
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn ffi_validate_garbage_reports_issues() {
        let garbage = [0xDEu8; 100];
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_validate(garbage.as_ptr(), garbage.len(), ptr::null(), 0, &mut out);
        assert!(matches!(err, super::TgmError::Ok));
        let json = unsafe { slice::from_raw_parts(out.data, out.len) };
        let s = std::str::from_utf8(json).unwrap();
        assert!(s.contains("issues"));
        super::tgm_bytes_free(out);
    }

    // ── tgm_validate_file (previously zero tests) ──────────────────────

    #[test]
    fn ffi_validate_file_null_args() {
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_validate_file(ptr::null(), ptr::null(), 0, &mut out);
        assert!(matches!(err, super::TgmError::InvalidArg));
        let path = CString::new("/tmp/x.tgm").unwrap();
        let err = super::tgm_validate_file(path.as_ptr(), ptr::null(), 0, ptr::null_mut());
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn ffi_validate_file_nonexistent() {
        let path = CString::new("/nonexistent/path/to/missing-file.tgm").unwrap();
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_validate_file(path.as_ptr(), ptr::null(), 0, &mut out);
        assert!(matches!(err, super::TgmError::Io));
    }

    #[test]
    fn ffi_validate_file_valid_round_trip() {
        use std::io::Write;
        let encoded = ffi_encode_single_f32_tensor(&[1.0f32, 2.0, 3.0], "");
        let tmp = std::env::temp_dir().join(format!(
            "tensogram-ffi-validate-file-{}.tgm",
            std::process::id(),
        ));
        std::fs::File::create(&tmp)
            .unwrap()
            .write_all(&encoded)
            .unwrap();
        let path = CString::new(tmp.to_str().unwrap()).unwrap();
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_validate_file(path.as_ptr(), ptr::null(), 0, &mut out);
        assert!(matches!(err, super::TgmError::Ok));
        super::tgm_bytes_free(out);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn ffi_validate_file_invalid_level() {
        let path = CString::new("/tmp/dummy.tgm").unwrap();
        let level = CString::new("bogus").unwrap();
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_validate_file(path.as_ptr(), level.as_ptr(), 0, &mut out);
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    // ── tgm_decode with verify_hash on tampered payload ───────────────

    #[test]
    #[ignore = "v3: hash moved to frame footer — re-enable in phase 6"]
    fn ffi_decode_verify_hash_on_tampered_payload() {
        let values = vec![1.0f32; 256];
        let encoded = ffi_encode_with_hash(&values);
        let mut tampered = encoded.clone();
        // Tamper around 75% into the message so we hit the payload region,
        // not frame headers / CBOR descriptors.
        let pos = (tampered.len() * 75) / 100;
        tampered[pos] ^= 0xFF;
        tampered[pos + 1] ^= 0xFF;
        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_decode(
            tampered.as_ptr(),
            tampered.len(),
            /* verify_hash */ 1,
            /* native_byte_order */ 0,
            /* threads */ 0,
            &mut msg,
        );
        assert!(matches!(err, super::TgmError::HashMismatch));
    }

    // ── tgm_object_data with null out_len pointer ─────────────────────

    #[test]
    fn ffi_object_data_null_out_len_no_crash() {
        let encoded = ffi_encode_single_f32_tensor(&[1.0f32], "");
        let mut msg: *mut super::TgmMessage = ptr::null_mut();
        let err = super::tgm_decode(
            encoded.as_ptr(),
            encoded.len(),
            /* verify_hash */ 0,
            /* native_byte_order */ 0,
            /* threads */ 0,
            &mut msg,
        );
        assert!(matches!(err, super::TgmError::Ok));
        // Calling with null out_len must not crash
        let data = super::tgm_object_data(msg, 0, ptr::null_mut());
        assert!(!data.is_null());
        super::tgm_message_free(msg);
    }

    // ── tgm_decode_range on a compression that lacks a block index ────

    #[test]
    fn ffi_decode_range_on_compressed_without_offsets() {
        // Encode with zstd compression which doesn't support range decode
        // without block offsets.
        let json = CString::new(
            r#"{"version":3,"descriptors":[{"type":"ntensor","ndim":1,"shape":[100],"strides":[1],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"zstd"}]}"#,
        )
        .unwrap();
        let data: Vec<u8> = vec![0u8; 400];
        let data_ptr: *const u8 = data.as_ptr();
        let data_len = data.len();
        let mut out = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = super::tgm_encode(
            json.as_ptr(),
            &data_ptr as *const *const u8,
            &data_len as *const usize,
            1,
            ptr::null(),
            /* threads */ 0,
            &mut out,
        );
        assert!(matches!(err, super::TgmError::Ok));
        let encoded = unsafe { slice::from_raw_parts(out.data, out.len) }.to_vec();
        super::tgm_bytes_free(out);

        // Attempt to range-decode: should fail because zstd has no block index.
        let range_offset: u64 = 10;
        let range_count: u64 = 20;
        let mut out_buf = super::TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let mut out_count: usize = 0;
        let err = super::tgm_decode_range(
            encoded.as_ptr(),
            encoded.len(),
            0,
            &range_offset as *const u64,
            &range_count as *const u64,
            1,
            /* verify_hash */ 0,
            /* native_byte_order */ 0,
            /* threads */ 0,
            /* join */ 1,
            &mut out_buf,
            &mut out_count,
        );
        assert!(!matches!(err, super::TgmError::Ok));
    }

    // ── tgm_simple_packing_compute_params edge cases ──

    #[test]
    fn ffi_simple_packing_null_values() {
        let mut ref_val: f64 = 0.0;
        let mut bsf: i32 = 0;
        let err =
            super::tgm_simple_packing_compute_params(ptr::null(), 0, 16, 0, &mut ref_val, &mut bsf);
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn ffi_simple_packing_null_out_ref() {
        let values: [f64; 3] = [1.0, 2.0, 3.0];
        let mut bsf: i32 = 0;
        let err = super::tgm_simple_packing_compute_params(
            values.as_ptr(),
            3,
            16,
            0,
            ptr::null_mut(),
            &mut bsf,
        );
        assert!(matches!(err, super::TgmError::InvalidArg));
    }

    #[test]
    fn ffi_simple_packing_null_out_bsf() {
        let values: [f64; 3] = [1.0, 2.0, 3.0];
        let mut ref_val: f64 = 0.0;
        let err = super::tgm_simple_packing_compute_params(
            values.as_ptr(),
            3,
            16,
            0,
            &mut ref_val,
            ptr::null_mut(),
        );
        assert!(matches!(err, super::TgmError::InvalidArg));
    }
}

/// Compute a hash of the given data.
/// Returns `TGM_ERROR_OK` on success, fills `out` with a `tgm_bytes_t`
/// containing the hex-encoded hash string (NOT null-terminated).
/// Free with `tgm_bytes_free`.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_compute_hash(
    data: *const u8,
    data_len: usize,
    algo: *const c_char,
    out: *mut TgmBytes,
) -> TgmError {
    if data.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let algorithm = if algo.is_null() {
        HashAlgorithm::Xxh3
    } else {
        let s = match unsafe { CStr::from_ptr(algo) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                set_last_error("invalid UTF-8 in algo");
                return TgmError::InvalidArg;
            }
        };
        match HashAlgorithm::parse(s) {
            Ok(a) => a,
            Err(e) => {
                set_last_error(&e.to_string());
                return TgmError::InvalidArg;
            }
        }
    };

    let input = unsafe { slice::from_raw_parts(data, data_len) };
    let hex = tensogram::hash::compute_hash(input, algorithm);
    // Rebuild via boxed slice to guarantee capacity == len for tgm_bytes_free.
    let mut bytes = hex.into_bytes().into_boxed_slice().into_vec();
    let result = TgmBytes {
        data: bytes.as_mut_ptr(),
        len: bytes.len(),
    };
    std::mem::forget(bytes);
    unsafe {
        *out = result;
    }
    TgmError::Ok
}

// ---------------------------------------------------------------------------
// Streaming encoder
// ---------------------------------------------------------------------------

/// Opaque handle for a streaming encoder that writes data objects progressively.
pub struct TgmStreamingEncoder {
    inner: Option<StreamingEncoder<std::io::BufWriter<std::fs::File>>>,
}

/// JSON used for streaming encoder creation — version + optional extra/base keys.
#[derive(serde::Deserialize)]
struct StreamingEncodeJson {
    version: u16,
    #[serde(default)]
    base: Vec<BTreeMap<String, serde_json::Value>>,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

/// Parse metadata JSON for the streaming encoder (no "descriptors" key).
fn parse_streaming_metadata_json(json_str: &str) -> Result<GlobalMetadata, String> {
    let parsed: StreamingEncodeJson = serde_json::from_str(json_str)
        .map_err(|e| format!("failed to parse metadata JSON: {e}"))?;

    let cbor_base: Vec<BTreeMap<String, ciborium::Value>> = parsed
        .base
        .into_iter()
        .map(|entry| {
            entry
                .into_iter()
                .map(|(k, v)| (k, json_to_cbor(v)))
                .collect()
        })
        .collect();

    // Validate: no _reserved_ keys in base entries (library-managed namespace)
    for (i, entry) in cbor_base.iter().enumerate() {
        if entry.contains_key(RESERVED_KEY) {
            return Err(format!(
                "base[{i}] must not contain '{RESERVED_KEY}' key — the encoder populates it"
            ));
        }
    }

    let cbor_extra: BTreeMap<String, ciborium::Value> = parsed
        .extra
        .into_iter()
        .map(|(k, v)| (k, json_to_cbor(v)))
        .collect();

    Ok(GlobalMetadata {
        version: parsed.version,
        base: cbor_base,
        extra: cbor_extra,
        ..Default::default()
    })
}

/// Create a streaming encoder writing to a file.
///
/// `metadata_json` must contain `"version"` key (and optional extra keys,
/// but NOT `"descriptors"`).
///
/// `hash_algo`: null-terminated string ("xxh3") or NULL for no hash.
///
/// On success fills `out` with a `TgmStreamingEncoder` handle.
/// Free with `tgm_streaming_encoder_free` or finalize with
/// `tgm_streaming_encoder_finish`.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_streaming_encoder_create(
    path: *const c_char,
    metadata_json: *const c_char,
    hash_algo: *const c_char,
    threads: u32,
    out: *mut *mut TgmStreamingEncoder,
) -> TgmError {
    if path.is_null() || metadata_json.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in path: {e}"));
            return TgmError::InvalidArg;
        }
    };

    let json_str = match unsafe { CStr::from_ptr(metadata_json) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in metadata_json: {e}"));
            return TgmError::InvalidArg;
        }
    };

    let global_metadata = match parse_streaming_metadata_json(json_str) {
        Ok(m) => m,
        Err(e) => {
            set_last_error(&e);
            return TgmError::Metadata;
        }
    };

    let hash_algorithm = if hash_algo.is_null() {
        None
    } else {
        let s = match unsafe { CStr::from_ptr(hash_algo) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                set_last_error("invalid UTF-8 in hash_algo");
                return TgmError::InvalidArg;
            }
        };
        match HashAlgorithm::parse(s) {
            Ok(a) => Some(a),
            Err(e) => {
                set_last_error(&e.to_string());
                return TgmError::InvalidArg;
            }
        }
    };

    let file = match std::fs::File::create(path_str) {
        Ok(f) => f,
        Err(e) => {
            set_last_error(&e.to_string());
            return TgmError::Io;
        }
    };

    let options = EncodeOptions {
        hash_algorithm,
        threads,
        ..Default::default()
    };
    let writer = std::io::BufWriter::new(file);

    match StreamingEncoder::new(writer, &global_metadata, &options) {
        Ok(enc) => {
            let handle = Box::new(TgmStreamingEncoder { inner: Some(enc) });
            unsafe {
                *out = Box::into_raw(handle);
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
    }
}

/// Write a PrecederMetadata frame for the next data object.
///
/// `metadata_json` is a JSON object with per-object metadata keys
/// (e.g. `{"mars": {"param": "2t"}, "units": "K"}`).  The keys
/// become `payload[0]` in a GlobalMetadata CBOR with empty `common`.
///
/// Must be followed by exactly one `tgm_streaming_encoder_write` call
/// before another preceder or `tgm_streaming_encoder_finish`.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_streaming_encoder_write_preceder(
    enc: *mut TgmStreamingEncoder,
    metadata_json: *const c_char,
) -> TgmError {
    if enc.is_null() || metadata_json.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let json_str = match unsafe { CStr::from_ptr(metadata_json) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in metadata_json: {e}"));
            return TgmError::InvalidArg;
        }
    };

    let map: BTreeMap<String, ciborium::Value> =
        match serde_json::from_str::<serde_json::Value>(json_str) {
            Ok(serde_json::Value::Object(obj)) => {
                obj.into_iter().map(|(k, v)| (k, json_to_cbor(v))).collect()
            }
            Ok(_) => {
                set_last_error("metadata_json must be a JSON object");
                return TgmError::Metadata;
            }
            Err(e) => {
                set_last_error(&format!("failed to parse metadata JSON: {e}"));
                return TgmError::Metadata;
            }
        };

    let encoder = unsafe { &mut *enc };
    match encoder.inner.as_mut() {
        Some(inner) => match inner.write_preceder(map) {
            Ok(()) => TgmError::Ok,
            Err(e) => {
                set_last_error(&e.to_string());
                to_error_code(&e)
            }
        },
        None => {
            set_last_error("streaming encoder already finished");
            TgmError::InvalidArg
        }
    }
}

/// Write a single data object to the streaming encoder.
///
/// `descriptor_json` is a JSON object with the descriptor fields
/// (type, ndim, shape, strides, dtype, byte_order, encoding, filter,
/// compression, etc.).
#[unsafe(no_mangle)]
pub extern "C" fn tgm_streaming_encoder_write(
    enc: *mut TgmStreamingEncoder,
    descriptor_json: *const c_char,
    data: *const u8,
    data_len: usize,
) -> TgmError {
    if enc.is_null() || descriptor_json.is_null() || data.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let json_str = match unsafe { CStr::from_ptr(descriptor_json) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in descriptor_json: {e}"));
            return TgmError::InvalidArg;
        }
    };

    let descriptor: DataObjectDescriptor = match serde_json::from_str(json_str) {
        Ok(d) => d,
        Err(e) => {
            set_last_error(&format!("failed to parse descriptor JSON: {e}"));
            return TgmError::Metadata;
        }
    };

    let data_slice = unsafe { slice::from_raw_parts(data, data_len) };
    let encoder = unsafe { &mut *enc };

    match encoder.inner.as_mut() {
        Some(inner) => match inner.write_object(&descriptor, data_slice) {
            Ok(()) => TgmError::Ok,
            Err(e) => {
                set_last_error(&e.to_string());
                to_error_code(&e)
            }
        },
        None => {
            set_last_error("streaming encoder already finished");
            TgmError::InvalidArg
        }
    }
}

/// Write a single pre-encoded data object to the streaming encoder.
///
/// Like `tgm_streaming_encoder_write`, but `data` must already be encoded
/// according to the descriptor's pipeline (`encoding` / `filter` /
/// `compression`). The library does not run the encoding pipeline — it
/// validates the descriptor's pipeline configuration and writes the bytes
/// as-is into a data object frame. The hash (if configured on the encoder)
/// is recomputed over the caller's bytes.
///
/// `descriptor_json`: same JSON schema as `tgm_streaming_encoder_write`.
///
/// For `szip` compression, callers SHOULD include `szip_block_offsets`
/// (bit offsets, not byte offsets) in the descriptor's params so that
/// `tgm_decode_range` can locate compressed block boundaries later.
/// Other pipeline params (e.g. `simple_packing` reference value, scale
/// factors) must also be present in the descriptor.
///
/// Any `hash` field embedded in the descriptor JSON is ignored — the
/// library always recomputes the hash from the caller's bytes.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_streaming_encoder_write_pre_encoded(
    enc: *mut TgmStreamingEncoder,
    descriptor_json: *const c_char,
    data: *const u8,
    data_len: usize,
) -> TgmError {
    if enc.is_null() || descriptor_json.is_null() || data.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let json_str = match unsafe { CStr::from_ptr(descriptor_json) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in descriptor_json: {e}"));
            return TgmError::InvalidArg;
        }
    };

    let descriptor: DataObjectDescriptor = match serde_json::from_str(json_str) {
        Ok(d) => d,
        Err(e) => {
            set_last_error(&format!("failed to parse descriptor JSON: {e}"));
            return TgmError::Metadata;
        }
    };

    let data_slice = unsafe { slice::from_raw_parts(data, data_len) };
    let encoder = unsafe { &mut *enc };

    match encoder.inner.as_mut() {
        Some(inner) => match inner.write_object_pre_encoded(&descriptor, data_slice) {
            Ok(()) => TgmError::Ok,
            Err(e) => {
                set_last_error(&e.to_string());
                to_error_code(&e)
            }
        },
        None => {
            set_last_error("streaming encoder already finished");
            TgmError::InvalidArg
        }
    }
}

/// Return the number of objects written so far.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_streaming_encoder_count(enc: *const TgmStreamingEncoder) -> usize {
    if enc.is_null() {
        return 0;
    }
    unsafe { (*enc).inner.as_ref().map(|e| e.object_count()).unwrap_or(0) }
}

/// Finalize the streaming encoder, writing footer and closing the file.
///
/// After calling this, the handle is still valid but empty — the caller
/// must still call `tgm_streaming_encoder_free` to release it.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_streaming_encoder_finish(enc: *mut TgmStreamingEncoder) -> TgmError {
    if enc.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let encoder = unsafe { &mut *enc };
    match encoder.inner.take() {
        Some(inner) => match inner.finish() {
            Ok(_writer) => {
                // Writer is dropped, file is closed.
                // Do NOT free enc — caller must call tgm_streaming_encoder_free.
                TgmError::Ok
            }
            Err(e) => {
                set_last_error(&e.to_string());
                to_error_code(&e)
            }
        },
        None => {
            set_last_error("streaming encoder already finished");
            TgmError::InvalidArg
        }
    }
}

/// Free a streaming encoder without finalizing (abandons the output).
#[unsafe(no_mangle)]
pub extern "C" fn tgm_streaming_encoder_free(enc: *mut TgmStreamingEncoder) {
    if !enc.is_null() {
        unsafe {
            drop(Box::from_raw(enc));
        }
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Parse a C-string validation level into `ValidateOptions`.
fn parse_validate_options(
    level: *const c_char,
    check_canonical: i32,
) -> Result<ValidateOptions, (TgmError, String)> {
    let level_str = if level.is_null() {
        "default"
    } else {
        unsafe { CStr::from_ptr(level) }
            .to_str()
            .map_err(|_| (TgmError::InvalidArg, "invalid UTF-8 in level".to_string()))?
    };

    let (max_level, checksum_only) = match level_str {
        "quick" => (ValidationLevel::Structure, false),
        "default" => (ValidationLevel::Integrity, false),
        "checksum" => (ValidationLevel::Integrity, true),
        "full" => (ValidationLevel::Fidelity, false),
        other => {
            return Err((
                TgmError::InvalidArg,
                format!(
                    "unknown validation level: '{}', expected one of: quick, default, checksum, full",
                    other
                ),
            ));
        }
    };

    Ok(ValidateOptions {
        max_level,
        check_canonical: check_canonical != 0,
        checksum_only,
    })
}

/// Validate a single Tensogram message buffer.
///
/// `buf` / `buf_len`: the wire-format message bytes (single message).
///   `buf` may be NULL when `buf_len` is 0 (empty-buffer validation).
/// `level`: validation depth — null-terminated C string:
///   `"quick"` (structure only), `"default"` (up to hash check),
///   `"checksum"` (hash check, suppress structural warnings),
///   `"full"` (full decode + NaN/Inf scan). NULL defaults to `"default"`.
/// `check_canonical`: non-zero to check RFC 8949 CBOR key ordering.
/// `out`: receives UTF-8 JSON bytes describing the validation report.
///   Not NUL-terminated — use `out->len` for the byte count.
///   Free with `tgm_bytes_free`.
///
/// Returns `TGM_ERROR_OK` on success (even if the message has issues —
/// the issues are in the JSON report). Returns `TGM_ERROR_INVALID_ARG`
/// for argument validation failures (null pointers, invalid level string),
/// or `TGM_ERROR_ENCODING` if JSON serialization of the report fails.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_validate(
    buf: *const u8,
    buf_len: usize,
    level: *const c_char,
    check_canonical: i32,
    out: *mut TgmBytes,
) -> TgmError {
    if out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    // Allow buf=NULL when buf_len=0 (empty-buffer validation).
    if buf.is_null() && buf_len > 0 {
        set_last_error("null buf with non-zero buf_len");
        return TgmError::InvalidArg;
    }

    let options = match parse_validate_options(level, check_canonical) {
        Ok(o) => o,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };

    let data = if buf.is_null() {
        &[]
    } else {
        unsafe { slice::from_raw_parts(buf, buf_len) }
    };
    let report = validate_message(data, &options);

    match serde_json::to_vec(&report) {
        Ok(json_bytes) => {
            let mut json_bytes = json_bytes.into_boxed_slice().into_vec();
            let result = TgmBytes {
                data: json_bytes.as_mut_ptr(),
                len: json_bytes.len(),
            };
            std::mem::forget(json_bytes);
            unsafe {
                *out = result;
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&format!("JSON serialization failed: {e}"));
            TgmError::Encoding
        }
    }
}

/// Validate all messages in a `.tgm` file.
///
/// `path`: null-terminated UTF-8 path to the file.
/// `level`: validation depth (same as `tgm_validate`). NULL = `"default"`.
/// `check_canonical`: non-zero to check CBOR key ordering.
/// `out`: receives UTF-8 JSON bytes describing the file validation report.
///   Not NUL-terminated — use `out->len` for the byte count.
///   Free with `tgm_bytes_free`.
///
/// Returns `TGM_ERROR_OK` on success (issues are in the JSON).
/// Returns `TGM_ERROR_IO` if the file cannot be opened or read.
/// Returns `TGM_ERROR_INVALID_ARG` for null pointers or invalid level.
/// Returns `TGM_ERROR_ENCODING` if JSON serialization of the report fails.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_validate_file(
    path: *const c_char,
    level: *const c_char,
    check_canonical: i32,
    out: *mut TgmBytes,
) -> TgmError {
    if path.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in path: {e}"));
            return TgmError::InvalidArg;
        }
    };

    let options = match parse_validate_options(level, check_canonical) {
        Ok(o) => o,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };

    let report = match core_validate_file(Path::new(path_str), &options) {
        Ok(r) => r,
        Err(e) => {
            set_last_error(&e.to_string());
            return TgmError::Io;
        }
    };

    match serde_json::to_vec(&report) {
        Ok(json_bytes) => {
            let mut json_bytes = json_bytes.into_boxed_slice().into_vec();
            let result = TgmBytes {
                data: json_bytes.as_mut_ptr(),
                len: json_bytes.len(),
            };
            std::mem::forget(json_bytes);
            unsafe {
                *out = result;
            }
            TgmError::Ok
        }
        Err(e) => {
            set_last_error(&format!("JSON serialization failed: {e}"));
            TgmError::Encoding
        }
    }
}
