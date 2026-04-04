// FFI functions accept raw pointers by design — callers are responsible for
// validity. Marking every extern "C" fn as `unsafe` would be correct but
// makes cbindgen emit ugly signatures with no benefit to C callers.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

//! Tensogram C FFI
//!
//! Exposes the tensogram-core library to C and C++ callers via opaque handles,
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
//! - `"version"` (integer, required): wire format version (2).
//! - `"descriptors"` (array, required): one entry per data object. Each entry
//!   merges tensor info and encoding pipeline info into a single object:
//!   `type`, `ndim`, `shape`, `strides`, `dtype`, `byte_order`, `encoding`,
//!   `filter`, `compression`. Additional keys are stored as params.
//! - Any other top-level keys (e.g. `"mars"`) are stored as global extra metadata.

use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::slice;

use tensogram_core::{
    decode, decode_metadata, decode_object, decode_range, encode, scan, DataObjectDescriptor,
    DecodeOptions, EncodeOptions, GlobalMetadata, HashAlgorithm, StreamingEncoder, TensogramError,
    TensogramFile,
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
#[no_mangle]
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

/// Free a byte buffer returned by `tgm_encode`.
#[no_mangle]
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
/// The `"version"` and `"descriptors"` keys are consumed; all remaining keys
/// are forwarded into `GlobalMetadata::extra` as CBOR values.
fn parse_encode_json(
    json_str: &str,
) -> Result<(GlobalMetadata, Vec<DataObjectDescriptor>), String> {
    let parsed: EncodeJson = serde_json::from_str(json_str)
        .map_err(|e| format!("failed to parse metadata JSON: {e}"))?;

    // "descriptors" is a known key we pull out; everything else in `extra` goes
    // into GlobalMetadata::extra as CBOR. The serde(flatten) on EncodeJson
    // will capture unknown fields there, but "descriptors" itself is already
    // pulled into the dedicated field and NOT included in extra.
    let cbor_extra: BTreeMap<String, ciborium::Value> = parsed
        .extra
        .into_iter()
        .map(|(k, v)| (k, json_to_cbor(v)))
        .collect();

    let global_metadata = GlobalMetadata {
        version: parsed.version,
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
                tensogram_core::ByteOrder::Big => "big",
                tensogram_core::ByteOrder::Little => "little",
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
    let hash_type_strings = objects
        .iter()
        .map(|(desc, _)| {
            desc.hash
                .as_ref()
                .map(|h| CString::new(h.hash_type.as_str()).unwrap_or_default())
        })
        .collect();
    let hash_value_strings = objects
        .iter()
        .map(|(desc, _)| {
            desc.hash
                .as_ref()
                .map(|h| CString::new(h.value.as_str()).unwrap_or_default())
        })
        .collect();

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
    let ptrs = slice::from_raw_parts(data_ptrs, num_objects);
    let lens = slice::from_raw_parts(data_lens, num_objects);
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
        .map(|(&p, &l)| slice::from_raw_parts(p, l))
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

    let data_slices = collect_data_slices(data_ptrs, data_lens, num_objects)?;
    let hash_algorithm = parse_hash_algo(hash_algo)?;
    let options = EncodeOptions { hash_algorithm };

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
#[no_mangle]
pub extern "C" fn tgm_encode(
    metadata_json: *const c_char,
    data_ptrs: *const *const u8,
    data_lens: *const usize,
    num_objects: usize,
    hash_algo: *const c_char,
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
        parse_encode_args(json_str, data_ptrs, data_lens, num_objects, hash_algo)
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
#[no_mangle]
pub extern "C" fn tgm_decode(
    buf: *const u8,
    buf_len: usize,
    verify_hash: i32,
    out: *mut *mut TgmMessage,
) -> TgmError {
    if buf.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let data = unsafe { slice::from_raw_parts(buf, buf_len) };
    let options = DecodeOptions {
        verify_hash: verify_hash != 0,
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
#[no_mangle]
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
#[no_mangle]
pub extern "C" fn tgm_decode_object(
    buf: *const u8,
    buf_len: usize,
    index: usize,
    verify_hash: i32,
    out: *mut *mut TgmMessage,
) -> TgmError {
    if buf.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let data = unsafe { slice::from_raw_parts(buf, buf_len) };
    let options = DecodeOptions {
        verify_hash: verify_hash != 0,
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

/// Decode a partial range from an uncompressed object.
///
/// `ranges_offsets` / `ranges_counts`: parallel arrays of (element_offset, element_count).
/// `num_ranges`: length of both arrays.
///
/// On success, fills `out` with a `TgmBytes` buffer of the extracted bytes.
#[no_mangle]
pub extern "C" fn tgm_decode_range(
    buf: *const u8,
    buf_len: usize,
    object_index: usize,
    ranges_offsets: *const u64,
    ranges_counts: *const u64,
    num_ranges: usize,
    verify_hash: i32,
    out: *mut TgmBytes,
) -> TgmError {
    if buf.is_null() || out.is_null() {
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

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

/// Scan a buffer for message boundaries.
///
/// Returns a `TgmScanResult` handle. Access entries with `tgm_scan_count`
/// and `tgm_scan_entry`. Free with `tgm_scan_free`.
#[no_mangle]
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
    if result.is_null() {
        None
    } else {
        Some(&*result)
    }
}

/// # Safety: caller must pass valid, non-null pointer from tgm_decode*.
unsafe fn as_msg(msg: *const TgmMessage) -> Option<&'static TgmMessage> {
    if msg.is_null() {
        None
    } else {
        Some(&*msg)
    }
}

/// Returns the number of messages found by `tgm_scan`.
#[no_mangle]
pub extern "C" fn tgm_scan_count(result: *const TgmScanResult) -> usize {
    unsafe { as_scan(result).map(|r| r.entries.len()).unwrap_or(0) }
}

#[no_mangle]
pub extern "C" fn tgm_scan_entry(result: *const TgmScanResult, index: usize) -> TgmScanEntry {
    let fallback = TgmScanEntry {
        offset: 0,
        length: 0,
    };
    unsafe {
        as_scan(result)
            .and_then(|r| r.entries.get(index).copied())
            .unwrap_or(fallback)
    }
}

/// Free a scan result handle.
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
pub extern "C" fn tgm_message_num_objects(msg: *const TgmMessage) -> usize {
    unsafe { as_msg(msg).map(|m| m.objects.len()).unwrap_or(0) }
}

/// Returns the number of decoded payload buffers.
/// Equivalent to `tgm_message_num_objects` — kept for ABI compatibility.
#[no_mangle]
pub extern "C" fn tgm_message_num_decoded(msg: *const TgmMessage) -> usize {
    unsafe { as_msg(msg).map(|m| m.objects.len()).unwrap_or(0) }
}

/// Returns the number of dimensions for object at index.
#[no_mangle]
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
#[no_mangle]
pub extern "C" fn tgm_object_shape(msg: *const TgmMessage, index: usize) -> *const u64 {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.objects.get(index))
            .map(|(desc, _)| desc.shape.as_ptr())
            .unwrap_or(ptr::null())
    }
}

/// Returns a pointer to the strides array. Length is `tgm_object_ndim()`.
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
pub extern "C" fn tgm_payload_encoding(msg: *const TgmMessage, index: usize) -> *const c_char {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.encoding_strings.get(index))
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    }
}

/// Returns 1 if the object descriptor has a hash, 0 otherwise.
#[no_mangle]
pub extern "C" fn tgm_payload_has_hash(msg: *const TgmMessage, index: usize) -> i32 {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.objects.get(index))
            .and_then(|(desc, _)| desc.hash.as_ref())
            .is_some() as i32
    }
}

/// Extract a metadata handle from a decoded message.
/// The metadata handle is independent — free it separately with `tgm_metadata_free`.
#[no_mangle]
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
#[no_mangle]
pub extern "C" fn tgm_object_type(msg: *const TgmMessage, index: usize) -> *const c_char {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.type_strings.get(index))
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    }
}

/// Returns the byte order string ("big" or "little"). Valid until message freed.
#[no_mangle]
pub extern "C" fn tgm_object_byte_order(msg: *const TgmMessage, index: usize) -> *const c_char {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.byte_order_strings.get(index))
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    }
}

/// Returns the filter string (e.g. "none", "shuffle"). Valid until message freed.
#[no_mangle]
pub extern "C" fn tgm_object_filter(msg: *const TgmMessage, index: usize) -> *const c_char {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.filter_strings.get(index))
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    }
}

/// Returns the compression string (e.g. "none", "zstd"). Valid until message freed.
#[no_mangle]
pub extern "C" fn tgm_object_compression(msg: *const TgmMessage, index: usize) -> *const c_char {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.compression_strings.get(index))
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    }
}

/// Returns the hash type string ("xxh3") or NULL if no hash. Valid until message freed.
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
pub extern "C" fn tgm_metadata_version(meta: *const TgmMetadata) -> u64 {
    if meta.is_null() {
        return 0;
    }
    unsafe { (*meta).global_metadata.version as u64 }
}

/// Returns the number of objects described in the global metadata.
///
/// In wire format v2, `GlobalMetadata` does not embed per-object descriptors,
/// so this function always returns 0. Use `tgm_message_num_objects` on a
/// decoded `TgmMessage` to get the actual object count.
#[no_mangle]
pub extern "C" fn tgm_metadata_num_objects(meta: *const TgmMetadata) -> usize {
    if meta.is_null() {
        return 0;
    }
    0
}

/// Look up a string value by dot-notation key (e.g. "mars.class").
/// Returns NULL if the key is not found or is not a string.
/// The pointer is valid until the metadata handle is freed.
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
pub extern "C" fn tgm_file_message_count(file: *mut TgmFile, out_count: *mut usize) -> TgmError {
    if file.is_null() || out_count.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let f = unsafe { &mut (*file).file };
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
#[no_mangle]
pub extern "C" fn tgm_file_decode_message(
    file: *mut TgmFile,
    index: usize,
    verify_hash: i32,
    out: *mut *mut TgmMessage,
) -> TgmError {
    if file.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let f = unsafe { &mut (*file).file };
    let options = DecodeOptions {
        verify_hash: verify_hash != 0,
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
#[no_mangle]
pub extern "C" fn tgm_file_read_message(
    file: *mut TgmFile,
    index: usize,
    out: *mut TgmBytes,
) -> TgmError {
    if file.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }

    let f = unsafe { &mut (*file).file };

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
#[no_mangle]
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
    let result = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(f.path())
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
#[no_mangle]
pub extern "C" fn tgm_file_path(file: *const TgmFile) -> *const c_char {
    if file.is_null() {
        return ptr::null();
    }
    unsafe { (*file).path_string.as_ptr() }
}

/// Encode and append a message to the file.
/// Same JSON schema as `tgm_encode` for `metadata_json`.
#[no_mangle]
pub extern "C" fn tgm_file_append(
    file: *mut TgmFile,
    metadata_json: *const c_char,
    data_ptrs: *const *const u8,
    data_lens: *const usize,
    num_objects: usize,
    hash_algo: *const c_char,
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
        parse_encode_args(json_str, data_ptrs, data_lens, num_objects, hash_algo)
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
#[no_mangle]
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

fn lookup_cbor_value<'a>(
    global_metadata: &'a GlobalMetadata,
    key: &str,
) -> Option<&'a ciborium::Value> {
    let parts: Vec<&str> = key.split('.').collect();

    if parts.len() == 1 {
        match parts[0] {
            "version" => return None, // use tgm_metadata_version instead
            k => return global_metadata.extra.get(k),
        }
    }

    if parts.len() == 2 {
        let (ns, field) = (parts[0], parts[1]);
        if let Some(ciborium::Value::Map(entries)) = global_metadata.extra.get(ns) {
            for (k, v) in entries {
                if let ciborium::Value::Text(k_str) = k {
                    if k_str == field {
                        return Some(v);
                    }
                }
            }
        }
    }

    None
}

fn lookup_string_key(global_metadata: &GlobalMetadata, key: &str) -> Option<String> {
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
            Some(n as i64)
        }
        _ => None,
    })
}

fn lookup_float_key(global_metadata: &GlobalMetadata, key: &str) -> Option<f64> {
    lookup_cbor_value(global_metadata, key).and_then(|v| match v {
        ciborium::Value::Float(f) => Some(*f),
        ciborium::Value::Integer(i) => {
            let n: i128 = (*i).into();
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
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
pub extern "C" fn tgm_buffer_iter_free(iter: *mut TgmBufferIter) {
    if !iter.is_null() {
        unsafe {
            drop(Box::from_raw(iter));
        }
    }
}

/// Opaque handle for iterating over messages in a file.
pub struct TgmFileIter {
    inner: tensogram_core::FileMessageIter,
}

/// Create a file message iterator from an open TgmFile.
///
/// Scans the file to locate message boundaries. The file handle remains
/// usable after this call.
#[no_mangle]
pub extern "C" fn tgm_file_iter_create(file: *mut TgmFile, out: *mut *mut TgmFileIter) -> TgmError {
    if file.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let f = unsafe { &mut (*file).file };
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
#[no_mangle]
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
#[no_mangle]
pub extern "C" fn tgm_file_iter_free(iter: *mut TgmFileIter) {
    if !iter.is_null() {
        unsafe {
            drop(Box::from_raw(iter));
        }
    }
}

/// Opaque handle for iterating over objects within a single message.
pub struct TgmObjectIter {
    inner: tensogram_core::ObjectIter,
    /// Global metadata parsed from the message header, cloned into each
    /// yielded `TgmMessage` to preserve the original version and extra fields.
    global_metadata: GlobalMetadata,
}

/// Create an object iterator from raw message bytes.
///
/// Parses metadata once, then decodes each object on demand when
/// `tgm_object_iter_next` is called. The global metadata from the
/// original message is preserved in each yielded `TgmMessage`.
#[no_mangle]
pub extern "C" fn tgm_object_iter_create(
    buf: *const u8,
    buf_len: usize,
    verify_hash: i32,
    out: *mut *mut TgmObjectIter,
) -> TgmError {
    if buf.is_null() || out.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let data = unsafe { slice::from_raw_parts(buf, buf_len) };
    let options = DecodeOptions {
        verify_hash: verify_hash != 0,
    };

    // Parse global metadata from the message header so we can attach it to
    // each yielded TgmMessage instead of fabricating a default.
    let global_metadata = decode_metadata(data).unwrap_or_default();

    match tensogram_core::objects(data, options) {
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
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
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
        _ => b"unknown error\0",
    };
    s.as_ptr() as *const c_char
}

// ---------------------------------------------------------------------------
// Hash utilities
// ---------------------------------------------------------------------------

/// Compute a hash of the given data.
/// Returns `TGM_ERROR_OK` on success, fills `out` with a `tgm_bytes_t`
/// containing the hex-encoded hash string (NOT null-terminated).
/// Free with `tgm_bytes_free`.
#[no_mangle]
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
    let hex = tensogram_core::hash::compute_hash(input, algorithm);
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

/// JSON used for streaming encoder creation — version + optional extra keys.
#[derive(serde::Deserialize)]
struct StreamingEncodeJson {
    version: u16,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

/// Parse metadata JSON for the streaming encoder (no "descriptors" key).
fn parse_streaming_metadata_json(json_str: &str) -> Result<GlobalMetadata, String> {
    let parsed: StreamingEncodeJson = serde_json::from_str(json_str)
        .map_err(|e| format!("failed to parse metadata JSON: {e}"))?;

    let cbor_extra: BTreeMap<String, ciborium::Value> = parsed
        .extra
        .into_iter()
        .map(|(k, v)| (k, json_to_cbor(v)))
        .collect();

    Ok(GlobalMetadata {
        version: parsed.version,
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
#[no_mangle]
pub extern "C" fn tgm_streaming_encoder_create(
    path: *const c_char,
    metadata_json: *const c_char,
    hash_algo: *const c_char,
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

    let options = EncodeOptions { hash_algorithm };
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

/// Write a single data object to the streaming encoder.
///
/// `descriptor_json` is a JSON object with the descriptor fields
/// (type, ndim, shape, strides, dtype, byte_order, encoding, filter,
/// compression, etc.).
#[no_mangle]
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

/// Return the number of objects written so far.
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
pub extern "C" fn tgm_streaming_encoder_free(enc: *mut TgmStreamingEncoder) {
    if !enc.is_null() {
        unsafe {
            drop(Box::from_raw(enc));
        }
    }
}
