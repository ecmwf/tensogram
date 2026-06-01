// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

//! Integration tests for the synchronous FFI core surface through the C ABI:
//! `tgm_encode` / `tgm_encode_with_options` / `tgm_decode` /
//! `tgm_decode_metadata` / `tgm_decode_object` / `tgm_decode_range`, the
//! streaming encoder, and `tgm_validate` / `tgm_validate_file` — happy and
//! error paths. The C/Python/Fortran suites exercise these heavily but run in
//! separate processes, so `cargo llvm-cov` never sees them; these Rust tests
//! close that gap.

use std::ffi::CString;
use std::ptr;

use tensogram_ffi::*;

fn cstring(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn native_bo() -> &'static str {
    if cfg!(target_endian = "little") {
        "little"
    } else {
        "big"
    }
}

/// `{"descriptors":[ <one float32 1-D object> ]}` envelope for `tgm_encode`.
fn encode_meta_f32(n: usize) -> CString {
    cstring(&format!(
        concat!(
            r#"{{"descriptors":[{{"type":"ntensor","ndim":1,"shape":[{n}],"#,
            r#""strides":[1],"dtype":"float32","byte_order":"{bo}","#,
            r#""encoding":"none","filter":"none","compression":"none"}}]}}"#
        ),
        n = n,
        bo = native_bo()
    ))
}

/// One bare float32 descriptor object (no envelope) for the streaming encoder.
fn stream_desc_f32(n: usize) -> CString {
    cstring(&format!(
        concat!(
            r#"{{"type":"ntensor","ndim":1,"shape":[{n}],"strides":[1],"#,
            r#""dtype":"float32","byte_order":"{bo}","encoding":"none","#,
            r#""filter":"none","compression":"none"}}"#
        ),
        n = n,
        bo = native_bo()
    ))
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(vals.len() * 4);
    for &x in vals {
        v.extend_from_slice(&x.to_ne_bytes());
    }
    v
}

/// Encode one float32 object; return a COPY of the wire bytes (the owned
/// `TgmBytes` is freed here so the test exercises `tgm_bytes_free` too).
fn encode_one(meta: &CString, data: &[u8]) -> Vec<u8> {
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_encode(
        meta.as_ptr(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        1,
        ptr::null(),
        0,
        &mut out,
    );
    assert!(matches!(err, TgmError::Ok), "encode failed: {err:?}");
    let bytes = unsafe { std::slice::from_raw_parts(out.data, out.len) }.to_vec();
    tgm_bytes_free(out);
    bytes
}

// ── tgm_encode / tgm_decode round-trip ─────────────────────────────────

#[test]
fn encode_decode_roundtrip_f32() {
    let meta = encode_meta_f32(4);
    let data = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let wire = encode_one(&meta, &data);
    assert!(!wire.is_empty());

    let mut msg: *mut TgmMessage = ptr::null_mut();
    let err = tgm_decode(wire.as_ptr(), wire.len(), 1, 0, 0, &mut msg);
    assert!(matches!(err, TgmError::Ok), "decode failed: {err:?}");
    assert!(!msg.is_null());
    assert_eq!(tgm_message_num_objects(msg), 1);

    let mut out_len = 0usize;
    let dptr = tgm_object_data(msg, 0, &mut out_len);
    assert!(!dptr.is_null());
    assert_eq!(out_len, data.len());
    let decoded = unsafe { std::slice::from_raw_parts(dptr, out_len) };
    assert_eq!(decoded, &data[..]);

    tgm_message_free(msg);
    // free on a null handle must be a harmless no-op
    tgm_message_free(ptr::null_mut());
}

// ── tgm_encode error paths ─────────────────────────────────────────────

#[test]
fn encode_null_metadata_is_invalid_arg() {
    let data = f32_bytes(&[1.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_encode(
        ptr::null(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        1,
        ptr::null(),
        0,
        &mut out,
    );
    assert!(matches!(err, TgmError::InvalidArg));
}

#[test]
fn encode_null_out_is_invalid_arg() {
    let meta = encode_meta_f32(1);
    let data = f32_bytes(&[1.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let err = tgm_encode(
        meta.as_ptr(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        1,
        ptr::null(),
        0,
        ptr::null_mut(),
    );
    assert!(matches!(err, TgmError::InvalidArg));
}

#[test]
fn encode_invalid_utf8_metadata_is_invalid_arg() {
    let bad = CString::new(vec![0xff_u8, 0xfe]).unwrap();
    let data = f32_bytes(&[1.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_encode(
        bad.as_ptr(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        1,
        ptr::null(),
        0,
        &mut out,
    );
    assert!(matches!(err, TgmError::InvalidArg));
}

#[test]
fn encode_invalid_json_metadata_errors() {
    let bad = cstring("{ not valid json ");
    let data = f32_bytes(&[1.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_encode(
        bad.as_ptr(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        1,
        ptr::null(),
        0,
        &mut out,
    );
    assert!(!matches!(err, TgmError::Ok));
}

#[test]
fn encode_nan_rejected_by_default() {
    let meta = encode_meta_f32(2);
    let data = f32_bytes(&[f32::NAN, 1.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_encode(
        meta.as_ptr(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        1,
        ptr::null(),
        0,
        &mut out,
    );
    assert!(
        matches!(err, TgmError::Encoding),
        "expected Encoding, got {err:?}"
    );
}

// ── tgm_encode_with_options ────────────────────────────────────────────

#[test]
fn encode_with_options_null_equals_default() {
    let meta = encode_meta_f32(3);
    let data = f32_bytes(&[1.0, 2.0, 3.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_encode_with_options(
        meta.as_ptr(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        1,
        ptr::null(),
        0,
        ptr::null(),
        &mut out,
    );
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    assert!(out.len > 0);
    tgm_bytes_free(out);
}

#[test]
fn encode_with_options_allow_nan_substitutes() {
    let meta = encode_meta_f32(2);
    let data = f32_bytes(&[f32::NAN, 1.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let opts = TgmEncodeMaskOptions {
        allow_nan: true,
        allow_inf: false,
        nan_mask_method: ptr::null(),
        pos_inf_mask_method: ptr::null(),
        neg_inf_mask_method: ptr::null(),
        small_mask_threshold_bytes: 0,
    };
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_encode_with_options(
        meta.as_ptr(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        1,
        ptr::null(),
        0,
        &opts,
        &mut out,
    );
    assert!(
        matches!(err, TgmError::Ok),
        "allow_nan encode failed: {err:?}"
    );
    assert!(out.len > 0);
    tgm_bytes_free(out);
}

// ── tgm_decode error paths ─────────────────────────────────────────────

#[test]
fn decode_null_out_is_invalid_arg() {
    let wire = encode_one(&encode_meta_f32(1), &f32_bytes(&[1.0]));
    let err = tgm_decode(wire.as_ptr(), wire.len(), 1, 0, 0, ptr::null_mut());
    assert!(matches!(err, TgmError::InvalidArg));
}

#[test]
fn decode_null_buf_nonzero_len_is_invalid_arg() {
    let mut msg: *mut TgmMessage = ptr::null_mut();
    let err = tgm_decode(ptr::null(), 16, 1, 0, 0, &mut msg);
    assert!(matches!(err, TgmError::InvalidArg));
}

#[test]
fn decode_garbage_is_error() {
    let garbage = vec![0xde_u8, 0xad, 0xbe, 0xef, 0, 0, 0, 0];
    let mut msg: *mut TgmMessage = ptr::null_mut();
    let err = tgm_decode(garbage.as_ptr(), garbage.len(), 1, 0, 0, &mut msg);
    assert!(!matches!(err, TgmError::Ok));
}

// ── tgm_decode_metadata ────────────────────────────────────────────────

#[test]
fn decode_metadata_roundtrip() {
    let wire = encode_one(&encode_meta_f32(4), &f32_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let mut meta: *mut TgmMetadata = ptr::null_mut();
    let err = tgm_decode_metadata(wire.as_ptr(), wire.len(), &mut meta);
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    assert!(!meta.is_null());
    tgm_metadata_free(meta);
    tgm_metadata_free(ptr::null_mut());
}

#[test]
fn decode_metadata_garbage_is_error() {
    let g = vec![0u8; 8];
    let mut meta: *mut TgmMetadata = ptr::null_mut();
    let err = tgm_decode_metadata(g.as_ptr(), g.len(), &mut meta);
    assert!(!matches!(err, TgmError::Ok));
}

// ── tgm_decode_object ──────────────────────────────────────────────────

#[test]
fn decode_object_by_index() {
    let wire = encode_one(&encode_meta_f32(4), &f32_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let mut msg: *mut TgmMessage = ptr::null_mut();
    let err = tgm_decode_object(wire.as_ptr(), wire.len(), 0, 1, 0, 0, &mut msg);
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    assert_eq!(tgm_message_num_objects(msg), 1);
    tgm_message_free(msg);
}

#[test]
fn decode_object_out_of_range_is_error() {
    let wire = encode_one(&encode_meta_f32(4), &f32_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let mut msg: *mut TgmMessage = ptr::null_mut();
    let err = tgm_decode_object(wire.as_ptr(), wire.len(), 99, 1, 0, 0, &mut msg);
    assert!(!matches!(err, TgmError::Ok));
}

// ── tgm_decode_range ───────────────────────────────────────────────────

#[test]
fn decode_range_subset() {
    let wire = encode_one(&encode_meta_f32(4), &f32_bytes(&[10.0, 20.0, 30.0, 40.0]));
    let offsets = [1u64];
    let counts = [2u64];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let mut out_count = 0usize;
    let err = tgm_decode_range(
        wire.as_ptr(),
        wire.len(),
        0,
        offsets.as_ptr(),
        counts.as_ptr(),
        1,
        1,
        0,
        1,
        &mut out,
        &mut out_count,
    );
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    assert!(out.len > 0);
    if !out.data.is_null() {
        tgm_bytes_free(out);
    }
}

#[test]
fn decode_range_null_out_is_invalid_arg() {
    let wire = encode_one(&encode_meta_f32(4), &f32_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let offsets = [0u64];
    let counts = [1u64];
    let mut out_count = 0usize;
    let err = tgm_decode_range(
        wire.as_ptr(),
        wire.len(),
        0,
        offsets.as_ptr(),
        counts.as_ptr(),
        1,
        1,
        0,
        0,
        ptr::null_mut(),
        &mut out_count,
    );
    assert!(matches!(err, TgmError::InvalidArg));
}

// ── streaming encoder ──────────────────────────────────────────────────

#[test]
fn streaming_encoder_lifecycle() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let meta = cstring("{}");
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    let err = tgm_streaming_encoder_create(path.as_ptr(), meta.as_ptr(), ptr::null(), 0, &mut enc);
    assert!(matches!(err, TgmError::Ok), "create: {err:?}");
    assert!(!enc.is_null());

    let desc = stream_desc_f32(4);
    let d0 = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let err = tgm_streaming_encoder_write(enc, desc.as_ptr(), d0.as_ptr(), d0.len());
    assert!(matches!(err, TgmError::Ok), "write0: {err:?}");
    let d1 = f32_bytes(&[5.0, 6.0, 7.0, 8.0]);
    let err = tgm_streaming_encoder_write(enc, desc.as_ptr(), d1.as_ptr(), d1.len());
    assert!(matches!(err, TgmError::Ok), "write1: {err:?}");

    assert_eq!(tgm_streaming_encoder_count(enc), 2);

    let err = tgm_streaming_encoder_finish(enc);
    assert!(matches!(err, TgmError::Ok), "finish: {err:?}");
    tgm_streaming_encoder_free(enc);

    let written = std::fs::metadata(f.path()).unwrap().len();
    assert!(written > 0);
}

#[test]
fn streaming_encoder_null_path_is_invalid_arg() {
    let meta = cstring("{}");
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    let err = tgm_streaming_encoder_create(ptr::null(), meta.as_ptr(), ptr::null(), 0, &mut enc);
    assert!(matches!(err, TgmError::InvalidArg));
}

#[test]
fn streaming_encoder_reserved_key_in_base_rejected() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    // `_reserved_` is the library-managed namespace; supplying it must fail.
    let meta = cstring(r#"{"base":[{"_reserved_":{"x":1}}]}"#);
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    let err = tgm_streaming_encoder_create(path.as_ptr(), meta.as_ptr(), ptr::null(), 0, &mut enc);
    assert!(!matches!(err, TgmError::Ok));
}

#[test]
fn streaming_encoder_unwritable_path_is_io_error() {
    let path = cstring("/nonexistent/tensogram/dir/out.tgm");
    let meta = cstring("{}");
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    let err = tgm_streaming_encoder_create(path.as_ptr(), meta.as_ptr(), ptr::null(), 0, &mut enc);
    assert!(!matches!(err, TgmError::Ok));
}

#[test]
fn streaming_encoder_invalid_json_rejected() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let meta = cstring("{ bad json ");
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    let err = tgm_streaming_encoder_create(path.as_ptr(), meta.as_ptr(), ptr::null(), 0, &mut enc);
    assert!(!matches!(err, TgmError::Ok));
}

#[test]
fn streaming_encoder_free_without_finish_is_noop() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let meta = cstring("{}");
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_streaming_encoder_create(path.as_ptr(), meta.as_ptr(), ptr::null(), 0, &mut enc),
        TgmError::Ok
    ));
    // Abandon without finishing, then a null-free no-op.
    tgm_streaming_encoder_free(enc);
    tgm_streaming_encoder_free(ptr::null_mut());
}

// ── tgm_validate / tgm_validate_file ──────────────────────────────────

#[test]
fn validate_good_message_reports() {
    let wire = encode_one(&encode_meta_f32(4), &f32_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let level = cstring("full");
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_validate(wire.as_ptr(), wire.len(), level.as_ptr(), 1, &mut out);
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    assert!(out.len > 0, "expected a JSON report");
    tgm_bytes_free(out);
}

#[test]
fn validate_garbage_does_not_crash() {
    let g = vec![0xde_u8, 0xad, 0xbe, 0xef];
    let level = cstring("quick");
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_validate(g.as_ptr(), g.len(), level.as_ptr(), 0, &mut out);
    // Either a report describing the failure, or an error code — both fine.
    assert!(!matches!(err, TgmError::Ok) || out.len > 0);
    if !out.data.is_null() {
        tgm_bytes_free(out);
    }
}

#[test]
fn validate_invalid_level_is_invalid_arg() {
    let wire = encode_one(&encode_meta_f32(1), &f32_bytes(&[1.0]));
    let level = cstring("bogus-level");
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_validate(wire.as_ptr(), wire.len(), level.as_ptr(), 0, &mut out);
    assert!(matches!(err, TgmError::InvalidArg));
}

#[test]
fn validate_null_out_is_invalid_arg() {
    let wire = encode_one(&encode_meta_f32(1), &f32_bytes(&[1.0]));
    let level = cstring("quick");
    let err = tgm_validate(
        wire.as_ptr(),
        wire.len(),
        level.as_ptr(),
        0,
        ptr::null_mut(),
    );
    assert!(matches!(err, TgmError::InvalidArg));
}

#[test]
fn validate_file_good() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let meta = cstring("{}");
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_streaming_encoder_create(path.as_ptr(), meta.as_ptr(), ptr::null(), 0, &mut enc),
        TgmError::Ok
    ));
    let desc = stream_desc_f32(4);
    let d = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    assert!(matches!(
        tgm_streaming_encoder_write(enc, desc.as_ptr(), d.as_ptr(), d.len()),
        TgmError::Ok
    ));
    assert!(matches!(tgm_streaming_encoder_finish(enc), TgmError::Ok));
    tgm_streaming_encoder_free(enc);

    let level = cstring("full");
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_validate_file(path.as_ptr(), level.as_ptr(), 0, &mut out);
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    assert!(out.len > 0);
    tgm_bytes_free(out);
}

#[test]
fn validate_file_missing_path_is_error() {
    let path = cstring("/nonexistent/tensogram/does/not/exist.tgm");
    let level = cstring("quick");
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_validate_file(path.as_ptr(), level.as_ptr(), 0, &mut out);
    assert!(!matches!(err, TgmError::Ok));
    if !out.data.is_null() {
        tgm_bytes_free(out);
    }
}

#[test]
fn validate_file_null_path_is_invalid_arg() {
    let level = cstring("quick");
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_validate_file(ptr::null(), level.as_ptr(), 0, &mut out);
    assert!(matches!(err, TgmError::InvalidArg));
}
