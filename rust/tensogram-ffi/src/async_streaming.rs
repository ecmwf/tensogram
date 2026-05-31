// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Async streaming-encoder FFI surface.  PR 3 of the cpp-async plan.
//!
//! Wraps [`tensogram::AsyncStreamingEncoder`] so C/C++ callers can
//! drive a producer that writes Tensogram messages asynchronously.
//!
//! In v1 the only sink is a local file (`tokio::fs::File`).  Object-
//! store backends are reserved for a follow-up; the FFI shape is
//! designed so they can slot in without C ABI changes — only an
//! additional URL-aware constructor.

#![cfg(feature = "async")]

use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Arc;

use tokio::sync::Mutex;

use tensogram::types::DataObjectDescriptor;
use tensogram::{AsyncStreamingEncoder, EncodeOptions};

use crate::async_core::{TaskResult, TgmAsyncTask, TgmCancellationToken, spawn_or_set_error};
use crate::{TgmError, set_last_error};

/// Opaque async streaming-encoder handle.  Internally wraps an
/// `AsyncStreamingEncoder<tokio::fs::File>` behind a `tokio::sync::Mutex`
/// so concurrent `write_object` calls are serialised (the underlying
/// AsyncWrite is naturally serial — concurrent calls would interleave
/// frame bytes).
pub struct TgmAsyncStreamingEncoder {
    inner: Arc<Mutex<Option<AsyncStreamingEncoder<tokio::fs::File>>>>,
    path_string: CString,
}

/// Borrowed pointer to the encoder's path string.  Valid for the
/// lifetime of the handle.  Returns NULL on a NULL handle.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_streaming_encoder_path(
    enc: *const TgmAsyncStreamingEncoder,
) -> *const c_char {
    if enc.is_null() {
        return std::ptr::null();
    }
    unsafe { (*enc).path_string.as_ptr() }
}

/// Free the encoder handle.
///
/// **Drop semantics:** the underlying encoder is held behind an
/// `Arc<Mutex<Option<...>>>`.  If a task is mid-flight (e.g. a
/// `write_object` whose returned `tgm_async_task_t*` has not been
/// joined and freed yet), the task holds its own `Arc` clone of the
/// inner state.  Calling `tgm_async_streaming_encoder_free` while a
/// task is in flight is therefore memory-safe: the inner encoder
/// stays alive until the last `Arc` (the task's) drops.
///
/// However: if the encoder is dropped without
/// `tgm_async_streaming_encoder_finish` having completed, the on-disk
/// file is **structurally invalid** (no footer frames, no postamble,
/// `total_length = 0`).  Validating readers will reject the file.
/// This matches the cancellation-mid-stream contract: operational
/// systems do not trust truncated `.tgm` files.  Callers who care about a clean file must
/// drive a successful `finish` task before freeing the encoder.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_streaming_encoder_free(enc: *mut TgmAsyncStreamingEncoder) {
    if !enc.is_null() {
        unsafe { drop(Box::from_raw(enc)) };
    }
}

/// Create an async streaming encoder writing to a local file.
///
/// Returns a task that resolves to a `tgm_async_streaming_encoder_t*`
/// (joinable via `tgm_async_task_join_async_streaming_encoder`).
///
/// `metadata_json` follows the same schema as the sync streaming
/// encoder (see `tgm_streaming_encoder_create`).  The async encoder
/// writes the preamble + header metadata frame to the sink before
/// the task resolves.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn tgm_async_streaming_encoder_create(
    path: *const c_char,
    metadata_json: *const c_char,
    hash_algo: *const c_char,
    threads: u32,
    cancel: *mut TgmCancellationToken,
    timeout_ms: u64,
    out_task: *mut *mut TgmAsyncTask,
) -> TgmError {
    if path.is_null() || metadata_json.is_null() || out_task.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s.to_string(),
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in path: {e}"));
            return TgmError::InvalidArg;
        }
    };
    let json_str = match unsafe { CStr::from_ptr(metadata_json) }.to_str() {
        Ok(s) => s.to_string(),
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in metadata_json: {e}"));
            return TgmError::InvalidArg;
        }
    };
    let global_metadata = match crate::parse_streaming_metadata_json(&json_str) {
        Ok(m) => m,
        Err(e) => {
            set_last_error(&e);
            return TgmError::Metadata;
        }
    };
    let hashing = match crate::parse_hash_algo(hash_algo) {
        Ok(b) => b,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };
    let cancel_ref = if cancel.is_null() {
        None
    } else {
        Some(unsafe { &*cancel })
    };

    let opts = EncodeOptions {
        hashing,
        threads,
        ..Default::default()
    };

    let path_for_task = path_str.clone();
    let fut = async move {
        let file = tokio::fs::File::create(&path_for_task)
            .await
            .map_err(tensogram::TensogramError::Io)?;
        let enc = AsyncStreamingEncoder::new(file, &global_metadata, &opts).await?;
        let path_string = CString::new(path_for_task.as_str()).unwrap_or_default();
        let handle = Box::new(TgmAsyncStreamingEncoder {
            inner: Arc::new(Mutex::new(Some(enc))),
            path_string,
        });
        Ok(TaskResult::AsyncStreamingEncoder(handle))
    };
    spawn_or_set_error(fut, cancel_ref, timeout_ms, out_task)
}

/// Selects which `AsyncStreamingEncoder` method to invoke from the
/// shared FFI scaffolding.  Avoids duplicating ~50 lines of input
/// parsing + future setup across the two `write_*` entry points.
#[derive(Clone, Copy)]
enum WriteKind {
    /// Encoder runs the full encoding pipeline on `data`.
    Object,
    /// `data` is treated as already-encoded bytes; pipeline is bypassed.
    PreEncoded,
}

/// Internal: shared scaffolding for the two `write_*` entry points.
///
/// Parses descriptor JSON + data slice, copies the data into an owned
/// buffer, then spawns a task that locks the encoder and dispatches
/// to the chosen inner method.
#[allow(clippy::too_many_arguments)]
fn write_object_dispatch(
    enc: *mut TgmAsyncStreamingEncoder,
    descriptor_json: *const c_char,
    data: *const u8,
    len: usize,
    cancel: *mut TgmCancellationToken,
    timeout_ms: u64,
    out_task: *mut *mut TgmAsyncTask,
    kind: WriteKind,
) -> TgmError {
    if enc.is_null() || descriptor_json.is_null() || data.is_null() || out_task.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let inner = unsafe { (*enc).inner.clone() };
    let json_str = match unsafe { CStr::from_ptr(descriptor_json) }.to_str() {
        Ok(s) => s.to_string(),
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in descriptor_json: {e}"));
            return TgmError::InvalidArg;
        }
    };
    let descriptor: DataObjectDescriptor = match serde_json::from_str(&json_str) {
        Ok(d) => d,
        Err(e) => {
            set_last_error(&format!("invalid descriptor JSON: {e}"));
            return TgmError::Metadata;
        }
    };
    // Copy data into an owned Vec<u8> so the async task owns it for
    // the duration of the write.  The C caller's buffer must remain
    // valid only until this function returns.
    let data_vec = unsafe { std::slice::from_raw_parts(data, len) }.to_vec();
    let cancel_ref = if cancel.is_null() {
        None
    } else {
        Some(unsafe { &*cancel })
    };

    let fut = async move {
        let mut guard = inner.lock().await;
        let enc = guard.as_mut().ok_or_else(|| {
            tensogram::TensogramError::Framing("encoder already finished".to_string())
        })?;
        match kind {
            WriteKind::Object => enc.write_object(&descriptor, &data_vec).await?,
            WriteKind::PreEncoded => enc.write_object_pre_encoded(&descriptor, &data_vec).await?,
        }
        Ok(TaskResult::Void)
    };
    spawn_or_set_error(fut, cancel_ref, timeout_ms, out_task)
}

/// Encode and append one data object to the stream.
///
/// `descriptor_json` is a JSON-serialised `DataObjectDescriptor`
/// matching the sync `tgm_streaming_encoder_write_object` schema.
/// `data` / `len` describe the raw element buffer; the encoder runs
/// the configured filter / compression / encoding pipeline on it.
///
/// The C caller's `data` buffer is **copied** before this function
/// returns, so the buffer may be reused immediately.  The returned
/// task resolves to `void` (joinable via
/// [`tgm_async_task_join_void`]).
///
/// Concurrent `write_*` calls on the same encoder are serialised
/// via an internal `tokio::sync::Mutex`; the underlying AsyncWrite
/// is naturally serial.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn tgm_async_streaming_encoder_write_object(
    enc: *mut TgmAsyncStreamingEncoder,
    descriptor_json: *const c_char,
    data: *const u8,
    len: usize,
    cancel: *mut TgmCancellationToken,
    timeout_ms: u64,
    out_task: *mut *mut TgmAsyncTask,
) -> TgmError {
    write_object_dispatch(
        enc,
        descriptor_json,
        data,
        len,
        cancel,
        timeout_ms,
        out_task,
        WriteKind::Object,
    )
}

/// Append one already-encoded data object to the stream, bypassing
/// the encoder's filter / compression / encoding pipeline.  `data`
/// is treated as the final on-disk payload bytes; `descriptor_json`
/// must describe the post-pipeline frame.
///
/// Useful for relaying frames between encoders without redundant
/// transcoding.  Other semantics (data copy, serialisation,
/// cancellation, void result) match
/// [`tgm_async_streaming_encoder_write_object`].
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn tgm_async_streaming_encoder_write_pre_encoded(
    enc: *mut TgmAsyncStreamingEncoder,
    descriptor_json: *const c_char,
    data: *const u8,
    len: usize,
    cancel: *mut TgmCancellationToken,
    timeout_ms: u64,
    out_task: *mut *mut TgmAsyncTask,
) -> TgmError {
    write_object_dispatch(
        enc,
        descriptor_json,
        data,
        len,
        cancel,
        timeout_ms,
        out_task,
        WriteKind::PreEncoded,
    )
}

/// Append a per-message preceder metadata frame.  `metadata_json`
/// is a flat JSON object (one level of nesting); keys become CBOR
/// map keys, values are converted to CBOR primitives.
///
/// Resolves to `void` (joinable via [`tgm_async_task_join_void`]).
/// Serialised against `write_*` via the same internal mutex.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_streaming_encoder_write_preceder(
    enc: *mut TgmAsyncStreamingEncoder,
    metadata_json: *const c_char,
    cancel: *mut TgmCancellationToken,
    timeout_ms: u64,
    out_task: *mut *mut TgmAsyncTask,
) -> TgmError {
    if enc.is_null() || metadata_json.is_null() || out_task.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let inner = unsafe { (*enc).inner.clone() };
    let json_str = match unsafe { CStr::from_ptr(metadata_json) }.to_str() {
        Ok(s) => s.to_string(),
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in metadata_json: {e}"));
            return TgmError::InvalidArg;
        }
    };
    // Parse JSON object → BTreeMap<String, ciborium::Value>.
    let value: serde_json::Value = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("invalid metadata JSON: {e}"));
            return TgmError::Metadata;
        }
    };
    let map = match preceder_json_to_cbor_map(value) {
        Ok(m) => m,
        Err(e) => {
            set_last_error(&e);
            return TgmError::Metadata;
        }
    };
    let cancel_ref = if cancel.is_null() {
        None
    } else {
        Some(unsafe { &*cancel })
    };

    let fut = async move {
        let mut guard = inner.lock().await;
        let enc = guard.as_mut().ok_or_else(|| {
            tensogram::TensogramError::Framing("encoder already finished".to_string())
        })?;
        enc.write_preceder(map).await?;
        Ok(TaskResult::Void)
    };
    spawn_or_set_error(fut, cancel_ref, timeout_ms, out_task)
}

/// Finalise the stream: flush pending frames, write the footer,
/// and (if `backfill = true`) seek back to update the preamble's
/// `total_length`.
///
/// Consumes the encoder's inner state — subsequent `write_*` /
/// `finish` calls on the same handle return
/// [`TensogramError::Framing`] (`encoder already finished`).  The
/// handle itself must still be released with
/// [`tgm_async_streaming_encoder_free`].
///
/// `backfill` requires that the underlying `tokio::fs::File`
/// supports seeks (true for local files, may not be true for
/// future remote sinks).  If a clean file is required and the
/// task is cancelled mid-finish, the on-disk file is structurally
/// invalid (see free's drop semantics).
#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_streaming_encoder_finish(
    enc: *mut TgmAsyncStreamingEncoder,
    backfill: bool,
    cancel: *mut TgmCancellationToken,
    timeout_ms: u64,
    out_task: *mut *mut TgmAsyncTask,
) -> TgmError {
    if enc.is_null() || out_task.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let inner = unsafe { (*enc).inner.clone() };
    let cancel_ref = if cancel.is_null() {
        None
    } else {
        Some(unsafe { &*cancel })
    };

    let fut = async move {
        let mut guard = inner.lock().await;
        let enc = guard.take().ok_or_else(|| {
            tensogram::TensogramError::Framing("encoder already finished".to_string())
        })?;
        if backfill {
            let _file = enc.finish_with_backfill().await?;
        } else {
            let _file = enc.finish().await?;
        }
        Ok(TaskResult::Void)
    };
    spawn_or_set_error(fut, cancel_ref, timeout_ms, out_task)
}

/// Status returned by `tgm_async_streaming_encoder_try_object_count`.
///
/// Distinguishes the three states the previous overloaded sentinel
/// (`usize::MAX`) collapsed: invalid handle, busy with another in-flight
/// FFI call, encoder already finished, and a valid count.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TgmObjectCountStatus {
    /// `*out_count` is valid.
    Ok = 0,
    /// `enc` was NULL.
    NullHandle = 1,
    /// The encoder is currently locked by another in-flight async
    /// operation (write/finish).  Try again after that operation
    /// completes.
    Busy = 2,
    /// `tgm_async_streaming_encoder_finish` has already consumed the
    /// encoder; subsequent calls to other methods will error too.
    Finished = 3,
}

/// Snapshot the encoder's object count without blocking.  Returns a
/// status describing whether `*out_count` is valid.
///
/// Use this in preference to the convenience accessor when you need
/// to distinguish "0 objects written so far" from "handle invalid"
/// or "encoder already finished".
#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_streaming_encoder_try_object_count(
    enc: *const TgmAsyncStreamingEncoder,
    out_count: *mut usize,
) -> TgmObjectCountStatus {
    if enc.is_null() || out_count.is_null() {
        return TgmObjectCountStatus::NullHandle;
    }
    let inner = unsafe { &*enc }.inner.clone();
    match inner.try_lock() {
        Ok(g) => match g.as_ref() {
            Some(e) => {
                unsafe { *out_count = e.object_count() };
                TgmObjectCountStatus::Ok
            }
            None => TgmObjectCountStatus::Finished,
        },
        Err(_) => TgmObjectCountStatus::Busy,
    }
}

/// Convenience: object count as a single number, with the historical
/// `usize::MAX` sentinel covering all "not OK" outcomes (null handle,
/// busy, finished).  Callers that need to discriminate should use
/// [`tgm_async_streaming_encoder_try_object_count`] instead.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_streaming_encoder_object_count(
    enc: *const TgmAsyncStreamingEncoder,
) -> usize {
    let mut count: usize = 0;
    match tgm_async_streaming_encoder_try_object_count(enc, &mut count) {
        TgmObjectCountStatus::Ok => count,
        _ => usize::MAX,
    }
}

// ---------------------------------------------------------------------------
// Async streaming encoder join
// ---------------------------------------------------------------------------

/// Typed join for tasks returning a streaming-encoder handle.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_task_join_async_streaming_encoder(
    task: *mut TgmAsyncTask,
    out: *mut *mut TgmAsyncStreamingEncoder,
) -> TgmError {
    if out.is_null() {
        set_last_error("null out pointer");
        return TgmError::InvalidArg;
    }
    match crate::async_core::join_internal(task) {
        Ok(TaskResult::AsyncStreamingEncoder(h)) => {
            unsafe { *out = Box::into_raw(h) };
            TgmError::Ok
        }
        Ok(_) => {
            set_last_error("task result type mismatch (expected async_streaming_encoder)");
            TgmError::InvalidArg
        }
        Err(code) => code,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a JSON object (one level deep is sufficient for preceder
/// payloads in v1) into a CBOR map.  Nested arrays/objects are
/// preserved.
fn preceder_json_to_cbor_map(
    v: serde_json::Value,
) -> Result<BTreeMap<String, ciborium::Value>, String> {
    let obj = match v {
        serde_json::Value::Object(m) => m,
        _ => return Err("preceder metadata must be a JSON object".to_string()),
    };
    let mut out = BTreeMap::new();
    for (k, val) in obj {
        out.insert(k, json_to_cbor(val));
    }
    Ok(out)
}

fn json_to_cbor(v: serde_json::Value) -> ciborium::Value {
    use ciborium::value::Integer;
    match v {
        serde_json::Value::Null => ciborium::Value::Null,
        serde_json::Value::Bool(b) => ciborium::Value::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                ciborium::Value::Integer(Integer::from(i))
            } else if let Some(u) = n.as_u64() {
                // Unsigned integers above i64::MAX would lose precision via
                // the f64 fallback, so keep them as CBOR integers.
                ciborium::Value::Integer(Integer::from(u))
            } else if let Some(f) = n.as_f64() {
                ciborium::Value::Float(f)
            } else {
                ciborium::Value::Null
            }
        }
        serde_json::Value::String(s) => ciborium::Value::Text(s),
        serde_json::Value::Array(a) => {
            ciborium::Value::Array(a.into_iter().map(json_to_cbor).collect())
        }
        serde_json::Value::Object(m) => ciborium::Value::Map(
            m.into_iter()
                .map(|(k, v)| (ciborium::Value::Text(k), json_to_cbor(v)))
                .collect(),
        ),
    }
}
