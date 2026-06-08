// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

//! Integration tests for the async streaming encoder FFI surface (PR 3).

#![cfg(feature = "async")]

use std::ffi::CString;
use std::ptr;

use tensogram_ffi::async_core::*;
use tensogram_ffi::async_streaming::*;
use tensogram_ffi::*;

fn cstring(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn temp_path() -> tempfile::TempPath {
    tempfile::NamedTempFile::new().unwrap().into_temp_path()
}

#[test]
fn async_streaming_encoder_local_round_trip() {
    let path = temp_path();
    let path_str = cstring(path.to_str().unwrap());
    let meta = cstring(r#"{"base":[]}"#);
    let hash = cstring("xxh3");

    // Create encoder
    let mut create_task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_create(
        path_str.as_ptr(),
        meta.as_ptr(),
        hash.as_ptr(),
        0,
        ptr::null_mut(),
        0,
        &mut create_task,
    );
    assert!(matches!(err, TgmError::Ok));

    let mut enc: *mut TgmAsyncStreamingEncoder = ptr::null_mut();
    let err = tgm_async_task_join_async_streaming_encoder(create_task, &mut enc);
    assert!(matches!(err, TgmError::Ok));
    tgm_async_task_free(create_task);

    // Object count is 0 before any writes.
    assert_eq!(tgm_async_streaming_encoder_object_count(enc), 0);

    // Write one object
    let descriptor = cstring(
        r#"{"type":"ntensor","ndim":1,"shape":[4],"strides":[1],
            "dtype":"float32","byte_order":"little","encoding":"none",
            "filter":"none","compression":"none","params":{}}"#,
    );
    let data: Vec<u8> = (0..16).collect();
    let mut write_task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_write_object(
        enc,
        descriptor.as_ptr(),
        data.as_ptr(),
        data.len(),
        ptr::null_mut(),
        0,
        &mut write_task,
    );
    assert!(matches!(err, TgmError::Ok));

    let err = tgm_async_task_join_void(write_task);
    assert!(matches!(err, TgmError::Ok));
    tgm_async_task_free(write_task);

    assert_eq!(tgm_async_streaming_encoder_object_count(enc), 1);

    // Finish
    let mut finish_task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_finish(enc, false, ptr::null_mut(), 0, &mut finish_task);
    assert!(matches!(err, TgmError::Ok));

    let err = tgm_async_task_join_void(finish_task);
    assert!(matches!(err, TgmError::Ok));
    tgm_async_task_free(finish_task);

    tgm_async_streaming_encoder_free(enc);

    // Verify file decodes correctly via the sync Rust API.
    let bytes = std::fs::read(path.as_os_str()).unwrap();
    let (_, objects) = tensogram::decode(&bytes, &tensogram::DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].1, data);
}

#[test]
fn async_streaming_encoder_finish_with_backfill() {
    use std::io::Read;

    let path = temp_path();
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

    let descriptor = cstring(
        r#"{"type":"ntensor","ndim":1,"shape":[4],"strides":[1],
            "dtype":"float32","byte_order":"little","encoding":"none",
            "filter":"none","compression":"none","params":{}}"#,
    );
    let data: Vec<u8> = (0..16).collect();
    let mut wt: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_streaming_encoder_write_object(
        enc,
        descriptor.as_ptr(),
        data.as_ptr(),
        data.len(),
        ptr::null_mut(),
        0,
        &mut wt,
    );
    let _ = tgm_async_task_join_void(wt);
    tgm_async_task_free(wt);

    // Finish with backfill.
    let mut ft: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_streaming_encoder_finish(enc, true, ptr::null_mut(), 0, &mut ft);
    let err = tgm_async_task_join_void(ft);
    assert!(matches!(err, TgmError::Ok));
    tgm_async_task_free(ft);
    tgm_async_streaming_encoder_free(enc);

    // Verify total_length is patched in both preamble and postamble.
    let mut bytes = Vec::new();
    std::fs::File::open(path.as_os_str())
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    let mut pre = [0u8; 8];
    pre.copy_from_slice(&bytes[16..24]);
    let preamble_total = u64::from_be_bytes(pre);
    assert_eq!(preamble_total, bytes.len() as u64);
    let end = bytes.len();
    let mut post = [0u8; 8];
    post.copy_from_slice(&bytes[end - 16..end - 8]);
    let postamble_total = u64::from_be_bytes(post);
    assert_eq!(postamble_total, bytes.len() as u64);
}

#[test]
fn async_streaming_encoder_double_finish_errors() {
    let path = temp_path();
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

    let mut ft: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_streaming_encoder_finish(enc, false, ptr::null_mut(), 0, &mut ft);
    let _ = tgm_async_task_join_void(ft);
    tgm_async_task_free(ft);

    // Second finish must error.
    let mut ft2: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_finish(enc, false, ptr::null_mut(), 0, &mut ft2);
    assert!(matches!(err, TgmError::Ok)); // launch ok; result errors on join
    let err = tgm_async_task_join_void(ft2);
    assert!(!matches!(err, TgmError::Ok));
    tgm_async_task_free(ft2);

    tgm_async_streaming_encoder_free(enc);
}

#[test]
fn async_streaming_encoder_write_after_finish_errors() {
    let path = temp_path();
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

    let mut ft: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_streaming_encoder_finish(enc, false, ptr::null_mut(), 0, &mut ft);
    let _ = tgm_async_task_join_void(ft);
    tgm_async_task_free(ft);

    // Write after finish must error on join.
    let descriptor = cstring(
        r#"{"type":"ntensor","ndim":1,"shape":[4],"strides":[1],
            "dtype":"float32","byte_order":"little","encoding":"none",
            "filter":"none","compression":"none","params":{}}"#,
    );
    let data = [0u8; 16];
    let mut wt: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_streaming_encoder_write_object(
        enc,
        descriptor.as_ptr(),
        data.as_ptr(),
        data.len(),
        ptr::null_mut(),
        0,
        &mut wt,
    );
    let err = tgm_async_task_join_void(wt);
    assert!(!matches!(err, TgmError::Ok));
    tgm_async_task_free(wt);

    tgm_async_streaming_encoder_free(enc);
}

#[test]
fn async_streaming_encoder_with_preceder() {
    let path = temp_path();
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

    // Write preceder + object
    let preceder = cstring(r#"{"step":7}"#);
    let mut pt: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_write_preceder(
        enc,
        preceder.as_ptr(),
        ptr::null_mut(),
        0,
        &mut pt,
    );
    assert!(matches!(err, TgmError::Ok));
    let err = tgm_async_task_join_void(pt);
    assert!(matches!(err, TgmError::Ok));
    tgm_async_task_free(pt);

    let descriptor = cstring(
        r#"{"type":"ntensor","ndim":1,"shape":[4],"strides":[1],
            "dtype":"float32","byte_order":"little","encoding":"none",
            "filter":"none","compression":"none","params":{}}"#,
    );
    let data = [0u8; 16];
    let mut wt: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_streaming_encoder_write_object(
        enc,
        descriptor.as_ptr(),
        data.as_ptr(),
        data.len(),
        ptr::null_mut(),
        0,
        &mut wt,
    );
    let _ = tgm_async_task_join_void(wt);
    tgm_async_task_free(wt);

    let mut ft: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_streaming_encoder_finish(enc, false, ptr::null_mut(), 0, &mut ft);
    let _ = tgm_async_task_join_void(ft);
    tgm_async_task_free(ft);
    tgm_async_streaming_encoder_free(enc);

    // Decode and check the preceder payload.
    let bytes = std::fs::read(path.as_os_str()).unwrap();
    let (m, _) = tensogram::decode(&bytes, &tensogram::DecodeOptions::default()).unwrap();
    assert!(m.base[0].contains_key("step"));
}

#[test]
fn async_streaming_encoder_pre_encoded_round_trip() {
    let path = temp_path();
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

    let descriptor = cstring(
        r#"{"type":"ntensor","ndim":1,"shape":[8],"strides":[1],
            "dtype":"float32","byte_order":"little","encoding":"none",
            "filter":"none","compression":"none","params":{}}"#,
    );
    let data: Vec<u8> = (0..32).collect();
    let mut wt: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_write_pre_encoded(
        enc,
        descriptor.as_ptr(),
        data.as_ptr(),
        data.len(),
        ptr::null_mut(),
        0,
        &mut wt,
    );
    assert!(matches!(err, TgmError::Ok));
    let err = tgm_async_task_join_void(wt);
    assert!(matches!(err, TgmError::Ok));
    tgm_async_task_free(wt);

    let mut ft: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_streaming_encoder_finish(enc, false, ptr::null_mut(), 0, &mut ft);
    let _ = tgm_async_task_join_void(ft);
    tgm_async_task_free(ft);
    tgm_async_streaming_encoder_free(enc);

    let bytes = std::fs::read(path.as_os_str()).unwrap();
    let (_, objects) = tensogram::decode(&bytes, &tensogram::DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].1, data);
}

// ---------------------------------------------------------------------------
// Null-argument validation regression tests.
//
// The async streaming encoder entry points all start with a null-pointer
// guard — typically `if a.is_null() || b.is_null() || ... { return InvalidArg; }`.
// Without explicit coverage, cargo-mutants would replace those `||` with
// `&&` (so the early-return only fires when ALL pointers are null) or
// replace the entire function with `Default::default()` (i.e. `Ok`) and
// the tests above would still pass.  These tests pin down each null-arg
// branch.
// ---------------------------------------------------------------------------

/// Helper: dummy descriptor JSON for write tests.
fn dummy_descriptor() -> CString {
    cstring(
        r#"{"type":"ntensor","ndim":1,"shape":[1],"strides":[1],"dtype":"uint8","byte_order":"little","encoding":"none","filter":"none","compression":"none","params":{}}"#,
    )
}

/// Build a real encoder handle so we can also test "valid encoder, but
/// other args null" branches.
fn make_test_encoder() -> (tempfile::TempPath, *mut TgmAsyncStreamingEncoder) {
    let path = temp_path();
    let path_str = cstring(path.to_str().unwrap());
    let meta = cstring(r#"{"base":[]}"#);
    let mut create_task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_create(
        path_str.as_ptr(),
        meta.as_ptr(),
        ptr::null(),
        0,
        ptr::null_mut(),
        0,
        &mut create_task,
    );
    assert!(matches!(err, TgmError::Ok));
    let mut enc: *mut TgmAsyncStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_async_task_join_async_streaming_encoder(create_task, &mut enc),
        TgmError::Ok
    ));
    tgm_async_task_free(create_task);
    (path, enc)
}

#[test]
fn create_rejects_null_path() {
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let meta = cstring(r#"{"base":[]}"#);
    let err = tgm_async_streaming_encoder_create(
        ptr::null(),
        meta.as_ptr(),
        ptr::null(),
        0,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[test]
fn create_rejects_null_metadata() {
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let path = cstring("/tmp/never-created.tgm");
    let err = tgm_async_streaming_encoder_create(
        path.as_ptr(),
        ptr::null(),
        ptr::null(),
        0,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[test]
fn create_rejects_null_out_task() {
    let path = cstring("/tmp/never-created.tgm");
    let meta = cstring(r#"{"base":[]}"#);
    let err = tgm_async_streaming_encoder_create(
        path.as_ptr(),
        meta.as_ptr(),
        ptr::null(),
        0,
        ptr::null_mut(),
        0,
        ptr::null_mut(),
    );
    assert!(matches!(err, TgmError::InvalidArg));
}

#[test]
fn write_object_rejects_null_encoder() {
    let desc = dummy_descriptor();
    let data = [0u8; 1];
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_write_object(
        ptr::null_mut(),
        desc.as_ptr(),
        data.as_ptr(),
        data.len(),
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[test]
fn write_object_rejects_null_descriptor() {
    let (_p, enc) = make_test_encoder();
    let data = [0u8; 1];
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_write_object(
        enc,
        ptr::null(),
        data.as_ptr(),
        data.len(),
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
    tgm_async_streaming_encoder_free(enc);
}

#[test]
fn write_object_rejects_null_data() {
    let (_p, enc) = make_test_encoder();
    let desc = dummy_descriptor();
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_write_object(
        enc,
        desc.as_ptr(),
        ptr::null(),
        0,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
    tgm_async_streaming_encoder_free(enc);
}

#[test]
fn write_object_rejects_null_out_task() {
    let (_p, enc) = make_test_encoder();
    let desc = dummy_descriptor();
    let data = [0u8; 1];
    let err = tgm_async_streaming_encoder_write_object(
        enc,
        desc.as_ptr(),
        data.as_ptr(),
        data.len(),
        ptr::null_mut(),
        0,
        ptr::null_mut(),
    );
    assert!(matches!(err, TgmError::InvalidArg));
    tgm_async_streaming_encoder_free(enc);
}

#[test]
fn write_pre_encoded_rejects_null_encoder() {
    let desc = dummy_descriptor();
    let data = [0u8; 1];
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_write_pre_encoded(
        ptr::null_mut(),
        desc.as_ptr(),
        data.as_ptr(),
        data.len(),
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[test]
fn write_pre_encoded_rejects_null_data() {
    let (_p, enc) = make_test_encoder();
    let desc = dummy_descriptor();
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_write_pre_encoded(
        enc,
        desc.as_ptr(),
        ptr::null(),
        0,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
    tgm_async_streaming_encoder_free(enc);
}

#[test]
fn write_preceder_rejects_null_encoder() {
    let meta = cstring(r#"{"k":"v"}"#);
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_write_preceder(
        ptr::null_mut(),
        meta.as_ptr(),
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[test]
fn write_preceder_rejects_null_metadata() {
    let (_p, enc) = make_test_encoder();
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err =
        tgm_async_streaming_encoder_write_preceder(enc, ptr::null(), ptr::null_mut(), 0, &mut task);
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
    tgm_async_streaming_encoder_free(enc);
}

#[test]
fn write_preceder_rejects_null_out_task() {
    let (_p, enc) = make_test_encoder();
    let meta = cstring(r#"{"k":"v"}"#);
    let err = tgm_async_streaming_encoder_write_preceder(
        enc,
        meta.as_ptr(),
        ptr::null_mut(),
        0,
        ptr::null_mut(),
    );
    assert!(matches!(err, TgmError::InvalidArg));
    tgm_async_streaming_encoder_free(enc);
}

#[test]
fn finish_rejects_null_encoder() {
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err =
        tgm_async_streaming_encoder_finish(ptr::null_mut(), false, ptr::null_mut(), 0, &mut task);
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[test]
fn finish_rejects_null_out_task() {
    let (_p, enc) = make_test_encoder();
    let err = tgm_async_streaming_encoder_finish(enc, false, ptr::null_mut(), 0, ptr::null_mut());
    assert!(matches!(err, TgmError::InvalidArg));
    tgm_async_streaming_encoder_free(enc);
}

// ---------------------------------------------------------------------------
// Accessor + non-null error-branch coverage.
// ---------------------------------------------------------------------------

#[test]
fn encoder_path_returns_path_and_null() {
    let (p, enc) = make_test_encoder();
    let ptr_path = tgm_async_streaming_encoder_path(enc);
    assert!(!ptr_path.is_null());
    let s = unsafe { std::ffi::CStr::from_ptr(ptr_path) }
        .to_str()
        .unwrap();
    assert_eq!(s, p.to_str().unwrap());
    tgm_async_streaming_encoder_free(enc);

    // NULL handle -> NULL.
    assert!(tgm_async_streaming_encoder_path(ptr::null()).is_null());
}

#[test]
fn encoder_free_null_is_noop() {
    tgm_async_streaming_encoder_free(ptr::null_mut());
}

#[test]
fn create_invalid_utf8_path_is_invalid_arg() {
    let bad = [0xff_u8, 0xfe, 0x00];
    let meta = cstring(r#"{"base":[]}"#);
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_create(
        bad.as_ptr() as *const std::os::raw::c_char,
        meta.as_ptr(),
        ptr::null(),
        0,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[test]
fn create_invalid_utf8_metadata_is_invalid_arg() {
    let path = cstring("/tmp/never.tgm");
    let bad = [0xff_u8, 0xfe, 0x00];
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_create(
        path.as_ptr(),
        bad.as_ptr() as *const std::os::raw::c_char,
        ptr::null(),
        0,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[test]
fn create_bad_metadata_is_metadata_error() {
    let path = cstring("/tmp/never.tgm");
    let bad = cstring("{ not json ");
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_create(
        path.as_ptr(),
        bad.as_ptr(),
        ptr::null(),
        0,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::Metadata));
    assert!(task.is_null());
}

#[test]
fn create_bad_hash_algo_is_invalid_arg() {
    let path = cstring("/tmp/never.tgm");
    let meta = cstring(r#"{"base":[]}"#);
    let algo = cstring("sha256");
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_create(
        path.as_ptr(),
        meta.as_ptr(),
        algo.as_ptr(),
        0,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
}

#[test]
fn create_io_error_surfaces_on_join() {
    // A nonexistent directory: file create fails inside the future.
    let path = cstring("/nonexistent/tensogram/async/stream.tgm");
    let meta = cstring(r#"{"base":[]}"#);
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_create(
        path.as_ptr(),
        meta.as_ptr(),
        ptr::null(),
        0,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::Ok)); // spawn ok; failure surfaces on join
    let mut enc: *mut TgmAsyncStreamingEncoder = ptr::null_mut();
    let jerr = tgm_async_task_join_async_streaming_encoder(task, &mut enc);
    assert!(!matches!(jerr, TgmError::Ok));
    tgm_async_task_free(task);
}

#[test]
fn write_object_invalid_utf8_descriptor_is_invalid_arg() {
    let (_p, enc) = make_test_encoder();
    let bad = [0xff_u8, 0xfe, 0x00];
    let data = [0u8; 1];
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_write_object(
        enc,
        bad.as_ptr() as *const std::os::raw::c_char,
        data.as_ptr(),
        data.len(),
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
    tgm_async_streaming_encoder_free(enc);
}

#[test]
fn write_object_bad_descriptor_json_is_metadata_error() {
    let (_p, enc) = make_test_encoder();
    let bad = cstring("{ not json ");
    let data = [0u8; 1];
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_write_object(
        enc,
        bad.as_ptr(),
        data.as_ptr(),
        data.len(),
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::Metadata));
    assert!(task.is_null());
    tgm_async_streaming_encoder_free(enc);
}

#[test]
fn write_pre_encoded_bad_descriptor_json_is_metadata_error() {
    let (_p, enc) = make_test_encoder();
    let bad = cstring("{ not json ");
    let data = [0u8; 1];
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_write_pre_encoded(
        enc,
        bad.as_ptr(),
        data.as_ptr(),
        data.len(),
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::Metadata));
    assert!(task.is_null());
    tgm_async_streaming_encoder_free(enc);
}

#[test]
fn write_preceder_invalid_utf8_metadata_is_invalid_arg() {
    let (_p, enc) = make_test_encoder();
    let bad = [0xff_u8, 0xfe, 0x00];
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_write_preceder(
        enc,
        bad.as_ptr() as *const std::os::raw::c_char,
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::InvalidArg));
    assert!(task.is_null());
    tgm_async_streaming_encoder_free(enc);
}

#[test]
fn write_preceder_bad_json_is_metadata_error() {
    let (_p, enc) = make_test_encoder();
    let bad = cstring("{ not json ");
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_write_preceder(
        enc,
        bad.as_ptr(),
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::Metadata));
    assert!(task.is_null());
    tgm_async_streaming_encoder_free(enc);
}

#[test]
fn write_preceder_non_object_is_metadata_error() {
    let (_p, enc) = make_test_encoder();
    // Valid JSON but an array, not an object.
    let arr = cstring("[1,2,3]");
    let mut task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_write_preceder(
        enc,
        arr.as_ptr(),
        ptr::null_mut(),
        0,
        &mut task,
    );
    assert!(matches!(err, TgmError::Metadata));
    assert!(task.is_null());
    tgm_async_streaming_encoder_free(enc);
}

#[test]
fn join_async_streaming_encoder_null_out_is_invalid_arg() {
    let path = temp_path();
    let path_str = cstring(path.to_str().unwrap());
    let meta = cstring(r#"{"base":[]}"#);
    let mut create_task: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_streaming_encoder_create(
        path_str.as_ptr(),
        meta.as_ptr(),
        ptr::null(),
        0,
        ptr::null_mut(),
        0,
        &mut create_task,
    );
    assert!(matches!(
        tgm_async_task_join_async_streaming_encoder(create_task, ptr::null_mut()),
        TgmError::InvalidArg
    ));
    // task not consumed; finish properly.
    let mut enc: *mut TgmAsyncStreamingEncoder = ptr::null_mut();
    let _ = tgm_async_task_join_async_streaming_encoder(create_task, &mut enc);
    tgm_async_task_free(create_task);
    if !enc.is_null() {
        tgm_async_streaming_encoder_free(enc);
    }
}

#[test]
fn join_async_streaming_encoder_null_task_is_invalid_arg() {
    let mut enc: *mut TgmAsyncStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_async_task_join_async_streaming_encoder(ptr::null_mut(), &mut enc),
        TgmError::InvalidArg
    ));
}

#[test]
fn join_async_streaming_encoder_type_mismatch_is_invalid_arg() {
    // A create task resolves to an encoder; a *finish* task resolves to
    // Void.  Joining the Void task as an encoder must surface a type
    // mismatch.
    let (_p, enc) = make_test_encoder();
    let mut ft: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_streaming_encoder_finish(enc, false, ptr::null_mut(), 0, &mut ft);
    // Wait for readiness so join resolves deterministically.
    while !tgm_async_task_is_ready(ft) {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let mut wrong: *mut TgmAsyncStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_async_task_join_async_streaming_encoder(ft, &mut wrong),
        TgmError::InvalidArg
    ));
    tgm_async_task_free(ft);
    tgm_async_streaming_encoder_free(enc);
}

#[test]
fn try_object_count_null_out_is_null_handle() {
    let (_p, enc) = make_test_encoder();
    let status = tgm_async_streaming_encoder_try_object_count(enc, ptr::null_mut());
    assert!(matches!(status, TgmObjectCountStatus::NullHandle));
    tgm_async_streaming_encoder_free(enc);
}

#[test]
fn object_count_convenience_sentinel_on_null() {
    assert_eq!(
        tgm_async_streaming_encoder_object_count(ptr::null()),
        usize::MAX
    );
}

#[test]
fn write_preceder_rich_value_types_round_trip() {
    // Exercises every json_to_cbor arm: null, bool, i64, u64-above-i64,
    // float, string, array, nested object.
    let path = temp_path();
    let path_str = cstring(path.to_str().unwrap());
    let meta = cstring(r#"{"base":[]}"#);
    let mut create_task: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_streaming_encoder_create(
        path_str.as_ptr(),
        meta.as_ptr(),
        ptr::null(),
        0,
        ptr::null_mut(),
        0,
        &mut create_task,
    );
    let mut enc: *mut TgmAsyncStreamingEncoder = ptr::null_mut();
    let _ = tgm_async_task_join_async_streaming_encoder(create_task, &mut enc);
    tgm_async_task_free(create_task);

    let preceder = cstring(
        r#"{"n":null,"b":true,"i":-7,"u":18446744073709551615,"f":3.5,
            "s":"txt","arr":[1,2.5,"x"],"obj":{"k":"v"}}"#,
    );
    let mut pt: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_write_preceder(
        enc,
        preceder.as_ptr(),
        ptr::null_mut(),
        0,
        &mut pt,
    );
    assert!(matches!(err, TgmError::Ok));
    assert!(matches!(tgm_async_task_join_void(pt), TgmError::Ok));
    tgm_async_task_free(pt);

    let descriptor = cstring(
        r#"{"type":"ntensor","ndim":1,"shape":[4],"strides":[1],
            "dtype":"float32","byte_order":"little","encoding":"none",
            "filter":"none","compression":"none","params":{}}"#,
    );
    let data = [0u8; 16];
    let mut wt: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_streaming_encoder_write_object(
        enc,
        descriptor.as_ptr(),
        data.as_ptr(),
        data.len(),
        ptr::null_mut(),
        0,
        &mut wt,
    );
    let _ = tgm_async_task_join_void(wt);
    tgm_async_task_free(wt);

    let mut ft: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_streaming_encoder_finish(enc, false, ptr::null_mut(), 0, &mut ft);
    let _ = tgm_async_task_join_void(ft);
    tgm_async_task_free(ft);
    tgm_async_streaming_encoder_free(enc);

    let bytes = std::fs::read(path.as_os_str()).unwrap();
    let (m, _) = tensogram::decode(&bytes, &tensogram::DecodeOptions::default()).unwrap();
    assert!(m.base[0].contains_key("arr"));
    assert!(m.base[0].contains_key("obj"));
}

/// Mutation-killer for the `threads` field in `EncodeOptions`
/// construction.  cargo-mutants mutates the struct expression to drop
/// the field; under that mutation the encoder still works because the
/// default is also reasonable, but with a non-trivial payload running
/// through the multi-threaded encode path we surface any difference.
///
/// The on-disk decode round-trip closes the related "function returns
/// Default::default()" mutation surface for `write_object` /
/// `write_pre_encoded` / `write_preceder` (the mutated function would
/// return Ok without writing, and the file would fail to decode).
#[test]
fn create_with_explicit_threads_round_trips() {
    let path = temp_path();
    let path_str = cstring(path.to_str().unwrap());
    let meta = cstring(r#"{"base":[]}"#);

    let mut create_task: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_create(
        path_str.as_ptr(),
        meta.as_ptr(),
        ptr::null(),
        4, // explicit threads > 0
        ptr::null_mut(),
        0,
        &mut create_task,
    );
    assert!(matches!(err, TgmError::Ok));
    assert!(!create_task.is_null());

    let mut enc: *mut TgmAsyncStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_async_task_join_async_streaming_encoder(create_task, &mut enc),
        TgmError::Ok
    ));
    tgm_async_task_free(create_task);

    // Drive a real write through the threaded path so the runtime
    // actually invokes the encoder's parallel stages.  Shape matches
    // data length so the encoder accepts the buffer without
    // complaining about size mismatch.
    let desc = cstring(
        r#"{"type":"ntensor","ndim":1,"shape":[1024],"strides":[1],"dtype":"uint8","byte_order":"little","encoding":"none","filter":"none","compression":"none","params":{}}"#,
    );
    let data = vec![0u8; 1024];
    let mut wt: *mut TgmAsyncTask = ptr::null_mut();
    let err = tgm_async_streaming_encoder_write_object(
        enc,
        desc.as_ptr(),
        data.as_ptr(),
        data.len(),
        ptr::null_mut(),
        0,
        &mut wt,
    );
    assert!(matches!(err, TgmError::Ok));
    assert!(matches!(tgm_async_task_join_void(wt), TgmError::Ok));
    tgm_async_task_free(wt);

    let mut ft: *mut TgmAsyncTask = ptr::null_mut();
    let _ = tgm_async_streaming_encoder_finish(enc, true, ptr::null_mut(), 0, &mut ft);
    assert!(matches!(tgm_async_task_join_void(ft), TgmError::Ok));
    tgm_async_task_free(ft);
    tgm_async_streaming_encoder_free(enc);

    // Verify the on-disk file decodes.  Guards against
    // "replace write_object -> TgmError with Default::default()".
    let bytes = std::fs::read(path.as_os_str()).unwrap();
    let (_, objects) = tensogram::decode(&bytes, &tensogram::DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].1.len(), 1024);
}
