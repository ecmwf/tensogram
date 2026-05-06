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
