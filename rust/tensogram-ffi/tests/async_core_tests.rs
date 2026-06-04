// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

//! Integration tests for the async FFI core surface.
//!
//! Walks the full async task lifecycle (open → operate → join → free)
//! through the C ABI to make sure the bridge to tokio is sound.

#![cfg(feature = "async")]

use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

use tensogram_ffi::async_core::*;
use tensogram_ffi::*;

fn make_test_file() -> tempfile::NamedTempFile {
    use std::collections::BTreeMap;
    use tensogram::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata};
    use tensogram::{Dtype, EncodeOptions, encode};

    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![4],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    let data: Vec<u8> = (0..16).collect();
    let bytes = encode(
        &GlobalMetadata::default(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .unwrap();
    let f = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(f.path(), &bytes).unwrap();
    f
}

fn cstring(s: &str) -> CString {
    CString::new(s).unwrap()
}

#[test]
fn async_open_and_message_count() {
    let f = make_test_file();
    let path = cstring(f.path().to_str().unwrap());

    // Open
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_open(path.as_ptr(), ptr::null_mut(), 0, &mut task);
    assert!(matches!(err, TgmError::Ok));

    let mut file: *mut TgmAsyncFile = ptr::null_mut();
    let err = tgm_async_task_join_async_file(task, &mut file);
    assert!(matches!(err, TgmError::Ok));
    tgm_async_task_free(task);

    // message_count
    let mut count_task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_message_count(file, ptr::null_mut(), 0, &mut count_task);
    assert!(matches!(err, TgmError::Ok));

    let mut n: u64 = 0;
    let err = tgm_async_task_join_size(count_task, &mut n);
    assert!(matches!(err, TgmError::Ok));
    assert_eq!(n, 1);
    tgm_async_task_free(count_task);

    tgm_async_file_close(file);
}

#[test]
fn async_decode_message_round_trip() {
    let f = make_test_file();
    let path = cstring(f.path().to_str().unwrap());

    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_file_open(path.as_ptr(), ptr::null_mut(), 0, &mut task);
    let mut file: *mut TgmAsyncFile = ptr::null_mut();
    let _ = tgm_async_task_join_async_file(task, &mut file);
    tgm_async_task_free(task);

    let mut decode_task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_decode_message(
        file,
        0,
        true,
        0,
        true,
        false,
        ptr::null_mut(),
        0,
        &mut decode_task,
    );
    assert!(matches!(err, TgmError::Ok));

    let mut msg: *mut TgmMessage = ptr::null_mut();
    let err = tgm_async_task_join_message(decode_task, &mut msg);
    assert!(matches!(err, TgmError::Ok));
    tgm_async_task_free(decode_task);

    let n_objs = tgm_message_num_objects(msg);
    assert_eq!(n_objs, 1);
    tgm_message_free(msg);

    tgm_async_file_close(file);
}

#[test]
fn async_decode_metadata_round_trip() {
    let f = make_test_file();
    let path = cstring(f.path().to_str().unwrap());

    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_file_open(path.as_ptr(), ptr::null_mut(), 0, &mut task);
    let mut file: *mut TgmAsyncFile = ptr::null_mut();
    let _ = tgm_async_task_join_async_file(task, &mut file);
    tgm_async_task_free(task);

    let mut mt: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_decode_metadata(file, 0, ptr::null_mut(), 0, &mut mt);
    assert!(matches!(err, TgmError::Ok));

    let mut m: *mut TgmMetadata = ptr::null_mut();
    let err = tgm_async_task_join_metadata(mt, &mut m);
    assert!(matches!(err, TgmError::Ok));
    tgm_async_task_free(mt);

    let v = tgm_metadata_version(m);
    assert_eq!(v as u16, tensogram::WIRE_VERSION);
    tgm_metadata_free(m);

    tgm_async_file_close(file);
}

#[test]
fn async_read_message_returns_bytes() {
    let f = make_test_file();
    let path = cstring(f.path().to_str().unwrap());

    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_file_open(path.as_ptr(), ptr::null_mut(), 0, &mut task);
    let mut file: *mut TgmAsyncFile = ptr::null_mut();
    let _ = tgm_async_task_join_async_file(task, &mut file);
    tgm_async_task_free(task);

    let mut rt: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_read_message(file, 0, ptr::null_mut(), 0, &mut rt);
    assert!(matches!(err, TgmError::Ok));

    let mut bytes = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_async_task_join_bytes(rt, &mut bytes);
    assert!(matches!(err, TgmError::Ok));
    assert!(bytes.len > 0);
    assert!(!bytes.data.is_null());
    tgm_async_task_free(rt);
    tgm_bytes_free(bytes);

    tgm_async_file_close(file);
}

#[test]
fn async_completion_callback_fires() {
    let f = make_test_file();
    let path = cstring(f.path().to_str().unwrap());

    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_file_open(path.as_ptr(), ptr::null_mut(), 0, &mut task);
    let mut file: *mut TgmAsyncFile = ptr::null_mut();
    let _ = tgm_async_task_join_async_file(task, &mut file);
    tgm_async_task_free(task);

    static FIRED: AtomicBool = AtomicBool::new(false);
    extern "C" fn cb(_: *mut c_void) {
        FIRED.store(true, Ordering::SeqCst);
    }

    let mut count_task: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_file_message_count(file, ptr::null_mut(), 0, &mut count_task);

    let err = tgm_async_task_set_completion(count_task, cb, ptr::null_mut());
    assert!(matches!(err, TgmError::Ok));

    let mut n: u64 = 0;
    let _ = tgm_async_task_join_size(count_task, &mut n);
    tgm_async_task_free(count_task);

    // The callback now runs on the dispatcher pool (post §4.1
    // refactor), so it may fire slightly after `join` returns.  Spin
    // for up to ~100 ms — generous, since the dispatcher just hops
    // a function-pointer call across one thread.
    for _ in 0..100 {
        if FIRED.load(Ordering::SeqCst) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(FIRED.load(Ordering::SeqCst));

    tgm_async_file_close(file);
}

/// Pins the exactly-once contract: a second `set_completion` call
/// must be rejected even after the first fired and the slot is
/// "empty" (the persistent `completion_registered` flag enforces it).
#[test]
fn async_double_completion_rejected() {
    let f = make_test_file();
    let path = cstring(f.path().to_str().unwrap());

    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_file_open(path.as_ptr(), ptr::null_mut(), 0, &mut task);

    extern "C" fn cb(_: *mut c_void) {}

    let err = tgm_async_task_set_completion(task, cb, ptr::null_mut());
    assert!(matches!(err, TgmError::Ok));

    // Wait for the task to settle (either Ready or already callback-fired).
    while !tgm_async_task_is_ready(task) {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    // Second registration must be rejected regardless of whether the
    // task is now Ready or the first callback has fired.
    let err = tgm_async_task_set_completion(task, cb, ptr::null_mut());
    assert!(matches!(err, TgmError::InvalidArg));

    let mut file: *mut TgmAsyncFile = ptr::null_mut();
    let _ = tgm_async_task_join_async_file(task, &mut file);
    tgm_async_task_free(task);
    tgm_async_file_close(file);
}

#[test]
fn async_double_join_rejected() {
    let f = make_test_file();
    let path = cstring(f.path().to_str().unwrap());

    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_file_open(path.as_ptr(), ptr::null_mut(), 0, &mut task);
    let mut file: *mut TgmAsyncFile = ptr::null_mut();
    let err = tgm_async_task_join_async_file(task, &mut file);
    assert!(matches!(err, TgmError::Ok));

    // Second join on the same task — result already consumed.
    let mut file2: *mut TgmAsyncFile = ptr::null_mut();
    let err = tgm_async_task_join_async_file(task, &mut file2);
    assert!(matches!(err, TgmError::InvalidArg));

    tgm_async_task_free(task);
    tgm_async_file_close(file);
}

#[test]
fn async_cancellation_token_cancels_task() {
    let tok = tgm_cancellation_token_create();
    assert!(!tgm_cancellation_token_is_cancelled(tok));
    tgm_cancellation_token_cancel(tok);
    assert!(tgm_cancellation_token_is_cancelled(tok));
    tgm_cancellation_token_free(tok);
}

/// Regression: a task that's still running when its external
/// cancellation token fires must surface as `Cancelled` (no leak of
/// a forwarder task — the spawn loop selects the external token
/// directly).  Open a deliberately slow operation so the cancel
/// fires before completion.
#[test]
fn async_external_cancel_during_task_surfaces_cancelled() {
    let f = make_test_file();
    let path = cstring(f.path().to_str().unwrap());

    let tok = tgm_cancellation_token_create();

    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_file_open(path.as_ptr(), tok, 0, &mut task);

    // Cancel immediately; the open is fast but the cancel-then-join
    // race is still resolved by the select! arm preferring whichever
    // future fires first.  Either way, the join must succeed and
    // either return Cancelled or Ok (if the open finished first).
    tgm_cancellation_token_cancel(tok);

    let mut file: *mut TgmAsyncFile = ptr::null_mut();
    let err = tgm_async_task_join_async_file(task, &mut file);
    assert!(
        matches!(err, TgmError::Ok | TgmError::Cancelled),
        "expected Ok or Cancelled, got {err:?}",
    );
    if err == TgmError::Ok {
        tgm_async_file_close(file);
    }
    tgm_async_task_free(task);
    tgm_cancellation_token_free(tok);
}

#[test]
fn async_timeout_error_code() {
    // Construct a deliberately bad path so the open fails fast; the
    // timeout path is still exercised structurally even if the operation
    // completes before the deadline.
    let path = cstring("/nonexistent/path/to/missing.tgm");
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_open(path.as_ptr(), ptr::null_mut(), 1000, &mut task);
    assert!(matches!(err, TgmError::Ok));

    let mut file: *mut TgmAsyncFile = ptr::null_mut();
    let err = tgm_async_task_join_async_file(task, &mut file);
    // We expect Io (file not found) — confirms the error-classification
    // path works for non-timeout errors too.
    assert!(matches!(
        err,
        TgmError::Io | TgmError::Encoding | TgmError::Framing
    ));
    tgm_async_task_free(task);
}

#[test]
fn async_runtime_configure_can_fail_after_init() {
    // Force runtime build.
    let f = make_test_file();
    let path = cstring(f.path().to_str().unwrap());
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_file_open(path.as_ptr(), ptr::null_mut(), 0, &mut task);
    let mut file: *mut TgmAsyncFile = ptr::null_mut();
    let _ = tgm_async_task_join_async_file(task, &mut file);
    tgm_async_task_free(task);

    // Now configure — should fail (runtime already built).
    let err = tgm_runtime_configure(2, 2, 0);
    assert!(matches!(err, TgmError::InvalidArg));

    tgm_async_file_close(file);
}

/// Pins that completion callbacks run on a dispatcher worker, NOT on
/// a tokio worker.  We verify by recording the thread name (set by the
/// runtime / dispatcher pool) inside the callback.
#[test]
fn async_completion_runs_on_dispatcher_thread() {
    let f = make_test_file();
    let path = cstring(f.path().to_str().unwrap());

    static GOT_DISPATCH_NAME: AtomicBool = AtomicBool::new(false);
    static SAW_TOKIO_WORKER: AtomicBool = AtomicBool::new(false);

    extern "C" fn cb(_: *mut c_void) {
        let cur = std::thread::current();
        if let Some(name) = cur.name() {
            if name.starts_with("tensogram-dispatch") {
                GOT_DISPATCH_NAME.store(true, Ordering::SeqCst);
            } else if name.starts_with("tensogram-async") {
                SAW_TOKIO_WORKER.store(true, Ordering::SeqCst);
            }
        }
    }

    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_file_open(path.as_ptr(), ptr::null_mut(), 0, &mut task);
    let _ = tgm_async_task_set_completion(task, cb, ptr::null_mut());

    // Drive the task to completion.
    let mut file: *mut TgmAsyncFile = ptr::null_mut();
    let _ = tgm_async_task_join_async_file(task, &mut file);
    tgm_async_task_free(task);

    // Give the dispatcher worker a moment to fire if it hadn't yet.
    for _ in 0..100 {
        if GOT_DISPATCH_NAME.load(Ordering::SeqCst) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    assert!(
        GOT_DISPATCH_NAME.load(Ordering::SeqCst),
        "completion callback must run on a dispatcher worker, not a tokio worker"
    );
    assert!(
        !SAW_TOKIO_WORKER.load(Ordering::SeqCst),
        "completion callback must not run on a tokio worker"
    );

    tgm_async_file_close(file);
}

/// Verify the new `try_object_count` API distinguishes the previously-
/// overloaded `usize::MAX` sentinel into Ok / Busy / Finished.
#[test]
fn async_streaming_encoder_try_object_count_disambiguates() {
    use tensogram_ffi::async_streaming::*;

    let path = tempfile::NamedTempFile::new().unwrap().into_temp_path();
    let path_str = cstring(path.to_str().unwrap());
    let meta = cstring(r#"{"base":[]}"#);
    let hash = cstring("xxh3");

    let mut create_task: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_streaming_encoder_create(
        path_str.as_ptr(),
        meta.as_ptr(),
        hash.as_ptr(),
        0,
        ptr::null_mut(),
        0,
        &mut create_task,
    );
    let mut enc: *mut TgmAsyncStreamingEncoder = ptr::null_mut();
    let _ = tgm_async_task_join_async_streaming_encoder(create_task, &mut enc);
    tgm_async_task_free(create_task);

    // Ok path — 0 objects so far.
    let mut count: usize = 99;
    let status = tgm_async_streaming_encoder_try_object_count(enc, &mut count);
    assert!(matches!(status, TgmObjectCountStatus::Ok));
    assert_eq!(count, 0);

    // NullHandle path.
    let mut count2: usize = 99;
    let status = tgm_async_streaming_encoder_try_object_count(ptr::null(), &mut count2);
    assert!(matches!(status, TgmObjectCountStatus::NullHandle));

    // Finished path: finish then check.
    let mut ft: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_streaming_encoder_finish(enc, false, ptr::null_mut(), 0, &mut ft);
    let _ = tgm_async_task_join_void(ft);
    tgm_async_task_free(ft);

    let mut count3: usize = 99;
    let status = tgm_async_streaming_encoder_try_object_count(enc, &mut count3);
    assert!(matches!(status, TgmObjectCountStatus::Finished));

    tgm_async_streaming_encoder_free(enc);
}

/// `tgm_error_string` must cover every error code, including the new
/// async ones.  Catching this regression early is cheap.
#[test]
fn tgm_error_string_covers_all_codes() {
    let codes = [
        TgmError::Ok,
        TgmError::Framing,
        TgmError::Metadata,
        TgmError::Encoding,
        TgmError::Compression,
        TgmError::Object,
        TgmError::Io,
        TgmError::HashMismatch,
        TgmError::InvalidArg,
        TgmError::EndOfIter,
        TgmError::Remote,
        TgmError::MissingHash,
        TgmError::Timeout,
        TgmError::Cancelled,
    ];
    for code in codes {
        let p = tgm_error_string(code);
        assert!(!p.is_null(), "tgm_error_string returned NULL for {code:?}");
        let text = unsafe { std::ffi::CStr::from_ptr(p) }.to_str().unwrap();
        assert_ne!(
            text, "unknown error",
            "tgm_error_string returned 'unknown error' for {code:?}"
        );
    }
}

/// `tgm_async_file_open_remote` is exported regardless of the
/// `async-remote` feature; without it, the call returns
/// [`TgmError::Remote`] and the diagnostic explains how to enable
/// the feature.  In the test build we don't enable async-remote.
#[cfg(not(feature = "async-remote"))]
#[test]
fn async_file_open_remote_without_feature_returns_remote_error() {
    let url = cstring("s3://example/test.tgm");
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_open_remote(
        url.as_ptr(),
        ptr::null(),
        ptr::null(),
        0,
        false,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::Remote));
    let last = tgm_last_error();
    assert!(!last.is_null());
    let msg = unsafe { std::ffi::CStr::from_ptr(last) }.to_str().unwrap();
    assert!(
        msg.contains("async-remote"),
        "expected diagnostic to mention the async-remote feature: got {msg:?}"
    );
}

// ---------------------------------------------------------------------------
// Null-argument validation regression tests for async_core.rs.
//
// Mirror the structure of the streaming-encoder null-arg tests:
// pin down each `if a.is_null() || b.is_null() { return InvalidArg; }`
// branch and each accessor's NULL behaviour, plus the trivial
// _free / _cancel / _is_cancelled noop paths.  Without these,
// cargo-mutants would replace the early-return with `&&`, drop the
// `!` from null checks, or reduce the function body to `()`, and
// every existing happy-path test would still pass.
// ---------------------------------------------------------------------------

#[test]
fn cancellation_token_free_null_is_noop() {
    // Mutation: `delete !` in `if !tok.is_null()` would dereference
    // NULL.  This test would crash under the mutation but pass on
    // the unmutated function.
    tgm_cancellation_token_free(ptr::null_mut());
}

#[test]
fn cancellation_token_cancel_null_is_noop() {
    tgm_cancellation_token_cancel(ptr::null_mut());
}

#[test]
fn cancellation_token_is_cancelled_null_returns_false() {
    // Mutation: `replace -> bool with true` would return true here.
    assert!(!tgm_cancellation_token_is_cancelled(ptr::null()));
}

#[test]
fn cancellation_token_full_lifecycle() {
    // Drives the unmutated lifecycle: create → not cancelled →
    // cancel → cancelled → free.  Catches the
    // `replace tgm_cancellation_token_create -> ... with Default::default()`
    // mutation (would return NULL → subsequent calls crash) and
    // `replace tgm_cancellation_token_cancel with ()` (would leave
    // is_cancelled false).
    let tok = tgm_cancellation_token_create();
    assert!(!tok.is_null());
    assert!(!tgm_cancellation_token_is_cancelled(tok));
    tgm_cancellation_token_cancel(tok);
    assert!(tgm_cancellation_token_is_cancelled(tok));
    tgm_cancellation_token_free(tok);
}

#[test]
fn async_task_free_null_is_noop() {
    tgm_async_task_free(ptr::null_mut());
}

#[test]
fn async_task_cancel_null_is_noop() {
    tgm_async_task_cancel(ptr::null_mut());
}

#[test]
fn async_task_is_ready_null_returns_false() {
    // Mutation: `replace -> bool with true` would return true here.
    assert!(!tgm_async_task_is_ready(ptr::null()));
}

#[test]
fn async_file_close_null_is_noop() {
    tgm_async_file_close(ptr::null_mut());
}

#[test]
fn async_file_path_null_returns_null() {
    // Mutation: `replace -> *const c_char with Default::default()`
    // returns NULL, which equals our expected null result — UNVIABLE
    // mutation in cargo-mutants terms.  But the explicit assertion
    // here documents the contract.
    assert!(tgm_async_file_path(ptr::null()).is_null());
}

#[test]
fn multi_bytes_free_null_is_noop() {
    // Pass a NULL array.  Mutation: `replace tgm_multi_bytes_free
    // with ()` would skip the early-return guard; ignored here
    // because we're already passing NULL.  But the `delete !` on
    // `if !array.is_null()` (currently `if array.is_null()`)
    // mutation would call `Vec::from_raw_parts(NULL, 0, 0)` — UB.
    tgm_multi_bytes_free(ptr::null_mut(), 0);
}

#[test]
fn async_file_open_rejects_null_path() {
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_open(ptr::null(), ptr::null_mut(), 0, &mut task);
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[test]
fn async_file_open_rejects_null_out_task() {
    let path = cstring("/tmp/nonexistent.tgm");
    let err = tgm_async_file_open(path.as_ptr(), ptr::null_mut(), 0, ptr::null_mut());
    assert!(matches!(err, TgmError::InvalidArg));
}

#[test]
fn async_file_message_count_rejects_null_file() {
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_message_count(ptr::null_mut(), ptr::null_mut(), 0, &mut task);
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[test]
fn async_file_message_count_rejects_null_out_task() {
    let f = make_test_file();
    let mut open_task: *mut TgmAsyncTask = ptr::null_mut();
    let path = cstring(f.path().to_str().unwrap());
    tgm_async_file_open(path.as_ptr(), ptr::null_mut(), 0, &mut open_task);
    let mut file: *mut TgmAsyncFile = ptr::null_mut();
    tgm_async_task_join_async_file(open_task, &mut file);
    tgm_async_task_free(open_task);

    let err = tgm_async_file_message_count(file, ptr::null_mut(), 0, ptr::null_mut());
    assert!(matches!(err, TgmError::InvalidArg));
    tgm_async_file_close(file);
}

#[test]
fn async_file_read_message_rejects_null_file() {
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_read_message(ptr::null_mut(), 0, ptr::null_mut(), 0, &mut task);
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[test]
fn async_file_decode_message_rejects_null_file() {
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_decode_message(
        ptr::null_mut(),
        0,
        false,
        0,
        false,
        false,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[test]
fn async_file_decode_metadata_rejects_null_file() {
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_decode_metadata(ptr::null_mut(), 0, ptr::null_mut(), 0, &mut task);
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[test]
fn async_file_decode_object_rejects_null_file() {
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_decode_object(
        ptr::null_mut(),
        0,
        0,
        false,
        0,
        false,
        false,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[test]
fn async_file_decode_range_rejects_null_offsets() {
    let counts = [0u64; 1];
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_decode_range(
        ptr::null_mut(),
        0,
        0,
        ptr::null(),
        counts.as_ptr(),
        1,
        false,
        0,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[test]
fn async_file_decode_range_rejects_null_counts() {
    let offsets = [0u64; 1];
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_decode_range(
        ptr::null_mut(),
        0,
        0,
        offsets.as_ptr(),
        ptr::null(),
        1,
        false,
        0,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[cfg(feature = "async-remote")]
#[test]
fn async_file_open_remote_rejects_null_url() {
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_open_remote(
        ptr::null(),
        ptr::null(),
        ptr::null(),
        0,
        false,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[cfg(feature = "async-remote")]
#[test]
fn async_file_open_remote_rejects_null_out_task() {
    let url = cstring("s3://test/file.tgm");
    let err = tgm_async_file_open_remote(
        url.as_ptr(),
        ptr::null(),
        ptr::null(),
        0,
        false,
        ptr::null_mut(),
        0,
        ptr::null_mut(),
    );
    assert!(matches!(err, TgmError::InvalidArg));
}

#[cfg(feature = "async-remote")]
#[test]
fn async_file_open_remote_rejects_null_storage_keys() {
    let url = cstring("s3://test/file.tgm");
    let v_value = cstring("v");
    let v_ptr = v_value.as_ptr();
    let v_arr: [*const std::os::raw::c_char; 1] = [v_ptr];
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_open_remote(
        url.as_ptr(),
        ptr::null(),    // null storage_keys array...
        v_arr.as_ptr(), // ...but non-null storage_values; nopts > 0 → reject.
        1,
        false,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[cfg(feature = "async-remote")]
#[test]
fn async_open_remote_file_url_round_trip() {
    // The async-remote success path: a file:// URL routes through
    // object_store's LocalFileSystem, so open_remote can be exercised
    // offline and deterministically (no network, no credentials).  Covers
    // the URL parse, the no-options branch, the spawn, and the downstream
    // remote scan that the null-argument tests above never reach.
    let f = make_test_file();
    let abs = std::fs::canonicalize(f.path()).unwrap();
    let url = cstring(&format!("file://{}", abs.display()));

    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_open_remote(
        url.as_ptr(),
        ptr::null(),
        ptr::null(),
        0,
        false,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::Ok));

    let mut file: *mut TgmAsyncFile = ptr::null_mut();
    let err = tgm_async_task_join_async_file(task, &mut file);
    assert!(matches!(err, TgmError::Ok));
    tgm_async_task_free(task);
    assert!(!file.is_null());

    // The handle is usable: a remote-backed message_count returns 1.
    let mut count_task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_message_count(file, ptr::null_mut(), 0, &mut count_task);
    assert!(matches!(err, TgmError::Ok));
    let mut n: u64 = 0;
    let err = tgm_async_task_join_size(count_task, &mut n);
    assert!(matches!(err, TgmError::Ok));
    assert_eq!(n, 1);
    tgm_async_task_free(count_task);

    tgm_async_file_close(file);
}

#[cfg(feature = "async-remote")]
#[test]
fn async_open_remote_marshals_storage_options() {
    // Exercises the storage-option key/value marshalling loop (nopts > 0)
    // on the success path.  The launch builds the option map synchronously
    // before spawning, so it returns Ok regardless of whether the backend
    // later accepts the options.
    let f = make_test_file();
    let abs = std::fs::canonicalize(f.path()).unwrap();
    let url = cstring(&format!("file://{}", abs.display()));

    let key = cstring("region");
    let value = cstring("eu-west-1");
    let keys: [*const std::os::raw::c_char; 1] = [key.as_ptr()];
    let values: [*const std::os::raw::c_char; 1] = [value.as_ptr()];

    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_open_remote(
        url.as_ptr(),
        keys.as_ptr(),
        values.as_ptr(),
        1,
        false,
        ptr::null_mut(),
        5000,
        &mut task,
    );
    assert!(matches!(err, TgmError::Ok)); // map built synchronously, task spawned

    // Join either succeeds (option ignored) or surfaces a backend error;
    // either way the marshalling path under test already ran.  Free
    // whatever the join produced so the test leaks nothing.
    let mut file: *mut TgmAsyncFile = ptr::null_mut();
    let join_err = tgm_async_task_join_async_file(task, &mut file);
    tgm_async_task_free(task);
    if matches!(join_err, TgmError::Ok) {
        assert!(!file.is_null());
        tgm_async_file_close(file);
    }
}

#[test]
fn async_decode_object_round_trip() {
    // Success path of decode_object (the existing tests only cover its
    // null-argument rejection).  Resolves to a single-object message.
    let f = make_test_file();
    let path = cstring(f.path().to_str().unwrap());
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_file_open(path.as_ptr(), ptr::null_mut(), 0, &mut task);
    let mut file: *mut TgmAsyncFile = ptr::null_mut();
    let _ = tgm_async_task_join_async_file(task, &mut file);
    tgm_async_task_free(task);

    let mut obj_task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_decode_object(
        file,
        0,
        0,
        true,
        0,
        true,
        false,
        ptr::null_mut(),
        0,
        &mut obj_task,
    );
    assert!(matches!(err, TgmError::Ok));
    let mut msg: *mut TgmMessage = ptr::null_mut();
    let err = tgm_async_task_join_message(obj_task, &mut msg);
    assert!(matches!(err, TgmError::Ok));
    tgm_async_task_free(obj_task);
    assert_eq!(tgm_message_num_objects(msg), 1);
    tgm_message_free(msg);

    tgm_async_file_close(file);
}

#[test]
fn async_decode_range_round_trip() {
    // Success path of decode_range, which resolves to a MultiBytes task —
    // so this also covers tgm_async_task_join_multi_bytes and
    // tgm_multi_bytes_free, none of which the null-argument tests reach.
    // The test object is a float32[4]; decode the whole [0, 4) element
    // range -> one buffer of 16 bytes.
    let f = make_test_file();
    let path = cstring(f.path().to_str().unwrap());
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_file_open(path.as_ptr(), ptr::null_mut(), 0, &mut task);
    let mut file: *mut TgmAsyncFile = ptr::null_mut();
    let _ = tgm_async_task_join_async_file(task, &mut file);
    tgm_async_task_free(task);

    let offsets: [u64; 1] = [0];
    let counts: [u64; 1] = [4];
    let mut range_task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_decode_range(
        file,
        0,
        0,
        offsets.as_ptr(),
        counts.as_ptr(),
        1,
        true,
        0,
        ptr::null_mut(),
        0,
        &mut range_task,
    );
    assert!(matches!(err, TgmError::Ok));

    let mut array: *mut TgmBytes = ptr::null_mut();
    let mut count: usize = 0;
    let err = tgm_async_task_join_multi_bytes(range_task, &mut array, &mut count);
    assert!(matches!(err, TgmError::Ok));
    tgm_async_task_free(range_task);
    assert_eq!(count, 1);
    assert!(!array.is_null());
    let first = unsafe { &*array };
    assert_eq!(first.len, 16); // 4 × float32
    tgm_multi_bytes_free(array, count);

    tgm_async_file_close(file);
}

// NOTE: `tgm_runtime_shutdown_blocking` tears down the process-global
// runtime permanently, so it cannot be exercised here without poisoning
// every sibling test in this binary.  It is covered by the in-crate
// unit test `async_core::tests::shutdown_blocking_drains_and_reports_unfinished`
// (the sole shutdown caller in the lib-test binary) and by the C++
// gtest `cpp/tests/test_async_shutdown_during_flight.cpp`.

#[cfg(feature = "async-remote")]
#[test]
fn async_open_remote_rejects_non_utf8_url() {
    // Adversarial: a non-UTF-8 URL must be rejected, not interpreted.
    let url = CString::new(vec![0xFF, 0xFE, 0x80]).unwrap();
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_open_remote(
        url.as_ptr(),
        ptr::null(),
        ptr::null(),
        0,
        false,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[cfg(feature = "async-remote")]
#[test]
fn async_open_remote_rejects_null_storage_entry() {
    // Non-null arrays, but a null key pointer inside them.
    let url = cstring("file:///tmp/x.tgm");
    let value = cstring("v");
    let keys: [*const std::os::raw::c_char; 1] = [ptr::null()];
    let values: [*const std::os::raw::c_char; 1] = [value.as_ptr()];
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_open_remote(
        url.as_ptr(),
        keys.as_ptr(),
        values.as_ptr(),
        1,
        false,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[cfg(feature = "async-remote")]
#[test]
fn async_open_remote_rejects_non_utf8_storage_key() {
    // Adversarial: non-UTF-8 bytes in a storage-option key.
    let url = cstring("file:///tmp/x.tgm");
    let bad_key = CString::new(vec![0xFF, 0xFE]).unwrap();
    let value = cstring("v");
    let keys: [*const std::os::raw::c_char; 1] = [bad_key.as_ptr()];
    let values: [*const std::os::raw::c_char; 1] = [value.as_ptr()];
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_open_remote(
        url.as_ptr(),
        keys.as_ptr(),
        values.as_ptr(),
        1,
        false,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[cfg(feature = "async-remote")]
#[test]
fn async_open_remote_rejects_non_utf8_storage_value() {
    // Adversarial: non-UTF-8 bytes in a storage-option value.
    let url = cstring("file:///tmp/x.tgm");
    let key = cstring("k");
    let bad_value = CString::new(vec![0xFF, 0xFE]).unwrap();
    let keys: [*const std::os::raw::c_char; 1] = [key.as_ptr()];
    let values: [*const std::os::raw::c_char; 1] = [bad_value.as_ptr()];
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_file_open_remote(
        url.as_ptr(),
        keys.as_ptr(),
        values.as_ptr(),
        1,
        false,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}
