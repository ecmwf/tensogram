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

use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::slice;

use tensogram_core::{
    decode, decode_metadata, decode_object, decode_range, encode, scan, DecodeOptions,
    EncodeOptions, HashAlgorithm, Metadata, TensogramError, TensogramFile,
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

/// Decoded message: metadata + decoded object payloads.
pub struct TgmMessage {
    metadata: Metadata,
    objects: Vec<Vec<u8>>,
    // Cached CStrings for accessor returns
    dtype_strings: Vec<CString>,
}

/// Metadata-only handle (no decoded payloads).
pub struct TgmMetadata {
    metadata: Metadata,
    // Cache for string accessors
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
// Encode
// ---------------------------------------------------------------------------

/// Encode a Tensogram message from JSON metadata and raw data slices.
///
/// `metadata_json`: null-terminated UTF-8 JSON string describing the message.
/// `data_ptrs` / `data_lens`: arrays of length `num_objects`, raw bytes per object.
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

    // Parse JSON → Metadata
    let metadata: Metadata = match serde_json::from_str(json_str) {
        Ok(m) => m,
        Err(e) => {
            set_last_error(&format!("failed to parse metadata JSON: {e}"));
            return TgmError::Metadata;
        }
    };

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
    let refs: Vec<&[u8]> = data_slices.to_vec();

    match encode(&metadata, &refs, &options) {
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

/// Decode a complete message (metadata + all object payloads).
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
        Ok((metadata, objects)) => {
            let dtype_strings: Vec<CString> = metadata
                .objects
                .iter()
                .map(|o| CString::new(o.dtype.to_string()).unwrap())
                .collect();
            let msg = Box::new(TgmMessage {
                metadata,
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

/// Decode only the metadata (no payload bytes are read).
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
        Ok(metadata) => {
            let m = Box::new(TgmMetadata {
                metadata,
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
/// one object (at index 0). The metadata covers the whole message.
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
        Ok((metadata, obj_bytes)) => {
            let dtype_strings: Vec<CString> = metadata
                .objects
                .iter()
                .map(|o| CString::new(o.dtype.to_string()).unwrap())
                .collect();
            let msg = Box::new(TgmMessage {
                metadata,
                objects: vec![obj_bytes],
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
    unsafe { as_msg(msg).map(|m| m.metadata.version).unwrap_or(0) }
}

#[no_mangle]
pub extern "C" fn tgm_message_num_objects(msg: *const TgmMessage) -> usize {
    unsafe { as_msg(msg).map(|m| m.metadata.objects.len()).unwrap_or(0) }
}

/// Returns the number of decoded payload buffers.
/// For tgm_decode this equals num_objects; for tgm_decode_object this is 1.
#[no_mangle]
pub extern "C" fn tgm_message_num_decoded(msg: *const TgmMessage) -> usize {
    unsafe { as_msg(msg).map(|m| m.objects.len()).unwrap_or(0) }
}

#[no_mangle]
pub extern "C" fn tgm_object_ndim(msg: *const TgmMessage, index: usize) -> u64 {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.metadata.objects.get(index))
            .map(|o| o.ndim)
            .unwrap_or(0)
    }
}

/// Returns a pointer to the shape array. Length is `tgm_object_ndim()`.
/// The pointer is valid until the message is freed.
#[no_mangle]
pub extern "C" fn tgm_object_shape(msg: *const TgmMessage, index: usize) -> *const u64 {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.metadata.objects.get(index))
            .map(|o| o.shape.as_ptr())
            .unwrap_or(ptr::null())
    }
}

/// Returns a pointer to the strides array. Length is `tgm_object_ndim()`.
#[no_mangle]
pub extern "C" fn tgm_object_strides(msg: *const TgmMessage, index: usize) -> *const u64 {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.metadata.objects.get(index))
            .map(|o| o.strides.as_ptr())
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
            Some(data) => {
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

/// Returns the encoding string for a payload descriptor (e.g. "none", "simple_packing").
#[no_mangle]
pub extern "C" fn tgm_payload_encoding(msg: *const TgmMessage, index: usize) -> *const c_char {
    static NONE: &[u8] = b"none\0";
    static SP: &[u8] = b"simple_packing\0";
    unsafe {
        match as_msg(msg).and_then(|m| m.metadata.payload.get(index)) {
            Some(p) => match p.encoding.as_str() {
                "none" => NONE.as_ptr() as *const c_char,
                "simple_packing" => SP.as_ptr() as *const c_char,
                _ => ptr::null(),
            },
            None => ptr::null(),
        }
    }
}

/// Returns 1 if the payload has a hash, 0 otherwise.
#[no_mangle]
pub extern "C" fn tgm_payload_has_hash(msg: *const TgmMessage, index: usize) -> i32 {
    unsafe {
        as_msg(msg)
            .and_then(|m| m.metadata.payload.get(index))
            .and_then(|p| p.hash.as_ref())
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
    unsafe { (*meta).metadata.version }
}

#[no_mangle]
pub extern "C" fn tgm_metadata_num_objects(meta: *const TgmMetadata) -> usize {
    if meta.is_null() {
        return 0;
    }
    unsafe { (*meta).metadata.objects.len() }
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
    let value = lookup_string_key(&m.metadata, key_str);

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
    lookup_int_key(&m.metadata, key_str).unwrap_or(default_val)
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
    lookup_float_key(&m.metadata, key_str).unwrap_or(default_val)
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
        Ok((metadata, objects)) => {
            let dtype_strings: Vec<CString> = metadata
                .objects
                .iter()
                .map(|o| CString::new(o.dtype.to_string()).unwrap())
                .collect();
            let msg = Box::new(TgmMessage {
                metadata,
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

fn lookup_cbor_value<'a>(metadata: &'a Metadata, key: &str) -> Option<&'a ciborium::Value> {
    let parts: Vec<&str> = key.split('.').collect();

    if parts.len() == 1 {
        match parts[0] {
            "version" => return None, // use tgm_metadata_version instead
            k => return metadata.extra.get(k),
        }
    }

    if parts.len() == 2 {
        let (ns, field) = (parts[0], parts[1]);
        // Check message-level extra
        if let Some(ciborium::Value::Map(entries)) = metadata.extra.get(ns) {
            for (k, v) in entries {
                if let ciborium::Value::Text(k_str) = k {
                    if k_str == field {
                        return Some(v);
                    }
                }
            }
        }
        // Check per-object extra (first match)
        for obj in &metadata.objects {
            if let Some(ciborium::Value::Map(entries)) = obj.extra.get(ns) {
                for (k, v) in entries {
                    if let ciborium::Value::Text(k_str) = k {
                        if k_str == field {
                            return Some(v);
                        }
                    }
                }
            }
        }
    }

    None
}

fn lookup_string_key(metadata: &Metadata, key: &str) -> Option<String> {
    // Handle special keys first
    if key == "version" {
        return Some(metadata.version.to_string());
    }

    lookup_cbor_value(metadata, key).and_then(|v| match v {
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

fn lookup_int_key(metadata: &Metadata, key: &str) -> Option<i64> {
    if key == "version" {
        return Some(metadata.version as i64);
    }

    lookup_cbor_value(metadata, key).and_then(|v| match v {
        ciborium::Value::Integer(i) => {
            let n: i128 = (*i).into();
            Some(n as i64)
        }
        _ => None,
    })
}

fn lookup_float_key(metadata: &Metadata, key: &str) -> Option<f64> {
    lookup_cbor_value(metadata, key).and_then(|v| match v {
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
        None => TgmError::Object, // sentinel: end of iteration
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
        None => TgmError::Object, // sentinel: end of iteration
        Some(Err(e)) => {
            set_last_error(&e.to_string());
            to_error_code(&e)
        }
        Some(Ok((descriptor, data))) => {
            let dtype_str = CString::new(descriptor.dtype.to_string()).unwrap_or_default();
            let metadata = Metadata {
                version: 1,
                objects: vec![descriptor],
                payload: vec![],
                extra: BTreeMap::new(),
            };
            let msg = Box::new(TgmMessage {
                metadata,
                objects: vec![data],
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
