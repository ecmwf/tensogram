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
    DecodeOptions, EncodeOptions, GlobalMetadata, HashAlgorithm, TensogramError, TensogramFile,
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
}

/// Scan result: array of (offset, length) pairs.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TgmScanEntry {
    pub offset: usize,
    pub length: usize,
}

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
    };

    Ok((global_metadata, parsed.descriptors))
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
/// `hash_algo`: null-terminated string ("xxh3", "sha1", "md5") or NULL for no hash.
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

    // Parse JSON → GlobalMetadata + Vec<DataObjectDescriptor>
    let (global_metadata, descriptors) = match parse_encode_json(json_str) {
        Ok(pair) => pair,
        Err(e) => {
            set_last_error(&e);
            return TgmError::Metadata;
        }
    };

    if descriptors.len() != num_objects {
        set_last_error(&format!(
            "descriptors array length {} does not match num_objects {}",
            descriptors.len(),
            num_objects
        ));
        return TgmError::InvalidArg;
    }

    // Collect data slices
    let data_slices: Vec<&[u8]> = if num_objects == 0 {
        vec![]
    } else {
        if data_ptrs.is_null() || data_lens.is_null() {
            set_last_error("null data_ptrs or data_lens");
            return TgmError::InvalidArg;
        }
        unsafe {
            let ptrs = slice::from_raw_parts(data_ptrs, num_objects);
            let lens = slice::from_raw_parts(data_lens, num_objects);
            ptrs.iter()
                .zip(lens.iter())
                .map(|(&p, &l)| slice::from_raw_parts(p, l))
                .collect()
        }
    };

    // Parse hash algorithm
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

    let options = EncodeOptions { hash_algorithm };

    // Build (descriptor, data) pairs for the encode API
    let pairs: Vec<(&DataObjectDescriptor, &[u8])> = descriptors
        .iter()
        .zip(data_slices.iter())
        .map(|(d, s)| (d, *s))
        .collect();

    match encode(&global_metadata, &pairs, &options) {
        Ok(mut bytes) => {
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
            let dtype_strings: Vec<CString> = objects
                .iter()
                .map(|(desc, _)| CString::new(desc.dtype.to_string()).unwrap_or_default())
                .collect();
            let msg = Box::new(TgmMessage {
                global_metadata,
                objects,
                dtype_strings,
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
            let dtype_str = CString::new(descriptor.dtype.to_string()).unwrap_or_default();
            let msg = Box::new(TgmMessage {
                global_metadata,
                objects: vec![(descriptor, obj_bytes)],
                dtype_strings: vec![dtype_str],
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
        Ok(mut bytes) => {
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
#[no_mangle]
pub extern "C" fn tgm_payload_encoding(msg: *const TgmMessage, index: usize) -> *const c_char {
    static NONE: &[u8] = b"none\0";
    static SP: &[u8] = b"simple_packing\0";
    unsafe {
        match as_msg(msg).and_then(|m| m.objects.get(index)) {
            Some((desc, _)) => match desc.encoding.as_str() {
                "none" => NONE.as_ptr() as *const c_char,
                "simple_packing" => SP.as_ptr() as *const c_char,
                _ => ptr::null(),
            },
            None => ptr::null(),
        }
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
            let handle = Box::new(TgmFile { file });
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
            let handle = Box::new(TgmFile { file });
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
            let dtype_strings: Vec<CString> = objects
                .iter()
                .map(|(desc, _)| CString::new(desc.dtype.to_string()).unwrap_or_default())
                .collect();
            let msg = Box::new(TgmMessage {
                global_metadata,
                objects,
                dtype_strings,
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
        Ok(mut bytes) => {
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
        Ok(()) => TgmError::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            TgmError::Io
        }
    }
}

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
        Some(Ok(mut bytes)) => {
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
}

/// Create an object iterator from raw message bytes.
///
/// Parses metadata once, then decodes each object on demand when
/// `tgm_object_iter_next` is called.
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
    match tensogram_core::objects(data, options) {
        Ok(inner) => {
            let iter = Box::new(TgmObjectIter { inner });
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
    let it = unsafe { &mut (*iter).inner };
    match it.next() {
        None => TgmError::EndOfIter,
        Some(Err(e)) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
        Some(Ok((descriptor, data))) => {
            let dtype_str = CString::new(descriptor.dtype.to_string()).unwrap_or_default();
            let global_metadata = GlobalMetadata {
                version: 2,
                extra: BTreeMap::new(),
            };
            let msg = Box::new(TgmMessage {
                global_metadata,
                objects: vec![(descriptor, data)],
                dtype_strings: vec![dtype_str],
            });
            unsafe {
                *out = Box::into_raw(msg);
            }
            TgmError::Ok
        }
    }
}

#[no_mangle]
pub extern "C" fn tgm_object_iter_free(iter: *mut TgmObjectIter) {
    if !iter.is_null() {
        unsafe {
            drop(Box::from_raw(iter));
        }
    }
}
