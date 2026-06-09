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

use std::ffi::{CStr, CString};
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
    let garbage = [0xde_u8, 0xad, 0xbe, 0xef, 0, 0, 0, 0];
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
    let g = [0u8; 8];
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
    let g = [0xde_u8, 0xad, 0xbe, 0xef];
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

// ── tgm_encode error: bad hash algo / num_objects mismatch / null data ──

#[test]
fn encode_bad_hash_algo_is_invalid_arg() {
    let meta = encode_meta_f32(1);
    let data = f32_bytes(&[1.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let algo = cstring("sha256");
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_encode(
        meta.as_ptr(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        1,
        algo.as_ptr(),
        0,
        &mut out,
    );
    assert!(matches!(err, TgmError::InvalidArg), "{err:?}");
}

#[test]
fn encode_invalid_utf8_hash_algo_is_invalid_arg() {
    let meta = encode_meta_f32(1);
    let data = f32_bytes(&[1.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let bad_algo = [0xff_u8, 0xfe, 0x00];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_encode(
        meta.as_ptr(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        1,
        bad_algo.as_ptr() as *const std::os::raw::c_char,
        0,
        &mut out,
    );
    assert!(matches!(err, TgmError::InvalidArg), "{err:?}");
}

#[test]
fn encode_num_objects_mismatch_is_invalid_arg() {
    // metadata describes 1 descriptor but we claim 2 objects.
    let meta = encode_meta_f32(1);
    let data = f32_bytes(&[1.0]);
    let ptrs = [data.as_ptr(), data.as_ptr()];
    let lens = [data.len(), data.len()];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_encode(
        meta.as_ptr(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        2,
        ptr::null(),
        0,
        &mut out,
    );
    assert!(matches!(err, TgmError::InvalidArg), "{err:?}");
}

#[test]
fn encode_null_data_ptrs_is_invalid_arg() {
    let meta = encode_meta_f32(1);
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_encode(
        meta.as_ptr(),
        ptr::null(),
        ptr::null(),
        1,
        ptr::null(),
        0,
        &mut out,
    );
    assert!(matches!(err, TgmError::InvalidArg), "{err:?}");
}

#[test]
fn encode_with_hash_xxh3_succeeds() {
    let meta = encode_meta_f32(2);
    let data = f32_bytes(&[1.0, 2.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let algo = cstring("xxh3");
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_encode(
        meta.as_ptr(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        1,
        algo.as_ptr(),
        0,
        &mut out,
    );
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    assert!(out.len > 0);
    tgm_bytes_free(out);
}

// ── tgm_encode_with_options error paths ────────────────────────────────

#[test]
fn encode_with_options_null_metadata_is_invalid_arg() {
    let data = f32_bytes(&[1.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_encode_with_options(
        ptr::null(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        1,
        ptr::null(),
        0,
        ptr::null(),
        &mut out,
    );
    assert!(matches!(err, TgmError::InvalidArg));
}

#[test]
fn encode_with_options_invalid_utf8_metadata_is_invalid_arg() {
    let bad = [0xff_u8, 0xfe, 0x00];
    let data = f32_bytes(&[1.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_encode_with_options(
        bad.as_ptr() as *const std::os::raw::c_char,
        ptrs.as_ptr(),
        lens.as_ptr(),
        1,
        ptr::null(),
        0,
        ptr::null(),
        &mut out,
    );
    assert!(matches!(err, TgmError::InvalidArg));
}

#[test]
fn encode_with_options_bad_mask_method_is_invalid_arg() {
    let meta = encode_meta_f32(2);
    let data = f32_bytes(&[f32::NAN, 1.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let bad_method = cstring("definitely-not-a-codec");
    let opts = TgmEncodeMaskOptions {
        allow_nan: true,
        allow_inf: false,
        nan_mask_method: bad_method.as_ptr(),
        pos_inf_mask_method: ptr::null(),
        neg_inf_mask_method: ptr::null(),
        small_mask_threshold_bytes: -1,
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
    assert!(matches!(err, TgmError::InvalidArg), "{err:?}");
}

#[test]
fn encode_with_options_explicit_mask_methods_succeeds() {
    let meta = encode_meta_f32(3);
    let data = f32_bytes(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let m = cstring("rle");
    let opts = TgmEncodeMaskOptions {
        allow_nan: true,
        allow_inf: true,
        nan_mask_method: m.as_ptr(),
        pos_inf_mask_method: m.as_ptr(),
        neg_inf_mask_method: m.as_ptr(),
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
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    assert!(out.len > 0);
    tgm_bytes_free(out);
}

// ── tgm_decode_with_options ────────────────────────────────────────────

#[test]
fn decode_with_options_null_default_roundtrip() {
    let wire = encode_one(&encode_meta_f32(4), &f32_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let mut msg: *mut TgmMessage = ptr::null_mut();
    let err = tgm_decode_with_options(wire.as_ptr(), wire.len(), 1, 0, 0, ptr::null(), &mut msg);
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    assert_eq!(tgm_message_num_objects(msg), 1);
    tgm_message_free(msg);
}

#[test]
fn decode_with_options_restore_non_finite_flag() {
    let wire = encode_one(&encode_meta_f32(2), &f32_bytes(&[1.0, 2.0]));
    let opts = TgmDecodeMaskOptions {
        restore_non_finite: false,
    };
    let mut msg: *mut TgmMessage = ptr::null_mut();
    let err = tgm_decode_with_options(wire.as_ptr(), wire.len(), 1, 0, 0, &opts, &mut msg);
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    tgm_message_free(msg);
}

#[test]
fn decode_with_options_null_out_is_invalid_arg() {
    let wire = encode_one(&encode_meta_f32(1), &f32_bytes(&[1.0]));
    let err = tgm_decode_with_options(
        wire.as_ptr(),
        wire.len(),
        1,
        0,
        0,
        ptr::null(),
        ptr::null_mut(),
    );
    assert!(matches!(err, TgmError::InvalidArg));
}

#[test]
fn decode_with_options_garbage_is_error() {
    let g = [0u8; 8];
    let mut msg: *mut TgmMessage = ptr::null_mut();
    let err = tgm_decode_with_options(g.as_ptr(), g.len(), 1, 0, 0, ptr::null(), &mut msg);
    assert!(!matches!(err, TgmError::Ok));
}

// ── tgm_encode_pre_encoded ─────────────────────────────────────────────

#[test]
fn encode_pre_encoded_roundtrip() {
    // For encoding "none" the raw bytes are already the wire payload.
    let meta = encode_meta_f32(4);
    let data = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_encode_pre_encoded(
        meta.as_ptr(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        1,
        ptr::null(),
        0,
        &mut out,
    );
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    assert!(out.len > 0);

    // Decode back and check.
    let wire = unsafe { std::slice::from_raw_parts(out.data, out.len) }.to_vec();
    tgm_bytes_free(out);
    let mut msg: *mut TgmMessage = ptr::null_mut();
    assert!(matches!(
        tgm_decode(wire.as_ptr(), wire.len(), 1, 0, 0, &mut msg),
        TgmError::Ok
    ));
    assert_eq!(tgm_message_num_objects(msg), 1);
    tgm_message_free(msg);
}

#[test]
fn encode_pre_encoded_null_metadata_is_invalid_arg() {
    let data = f32_bytes(&[1.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_encode_pre_encoded(
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
fn encode_pre_encoded_invalid_utf8_metadata_is_invalid_arg() {
    let bad = [0xff_u8, 0xfe, 0x00];
    let data = f32_bytes(&[1.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_encode_pre_encoded(
        bad.as_ptr() as *const std::os::raw::c_char,
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
fn encode_pre_encoded_bad_metadata_is_metadata_error() {
    let bad = cstring("{ not json ");
    let data = f32_bytes(&[1.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_encode_pre_encoded(
        bad.as_ptr(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        1,
        ptr::null(),
        0,
        &mut out,
    );
    assert!(matches!(err, TgmError::Metadata), "{err:?}");
}

// ── tgm_decode_object error paths ──────────────────────────────────────

#[test]
fn decode_object_null_out_is_invalid_arg() {
    let wire = encode_one(&encode_meta_f32(1), &f32_bytes(&[1.0]));
    let err = tgm_decode_object(wire.as_ptr(), wire.len(), 0, 1, 0, 0, ptr::null_mut());
    assert!(matches!(err, TgmError::InvalidArg));
}

// ── tgm_decode_metadata null out ───────────────────────────────────────

#[test]
fn decode_metadata_null_out_is_invalid_arg() {
    let wire = encode_one(&encode_meta_f32(1), &f32_bytes(&[1.0]));
    let err = tgm_decode_metadata(wire.as_ptr(), wire.len(), ptr::null_mut());
    assert!(matches!(err, TgmError::InvalidArg));
}

// ── tgm_decode_range error paths / join mode ───────────────────────────

#[test]
fn decode_range_join_mode() {
    let wire = encode_one(&encode_meta_f32(4), &f32_bytes(&[10.0, 20.0, 30.0, 40.0]));
    let offsets = [0u64, 2u64];
    let counts = [1u64, 2u64];
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
        2,
        1,
        0,
        1, // join
        &mut out,
        &mut out_count,
    );
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    assert_eq!(out_count, 1);
    if !out.data.is_null() {
        tgm_bytes_free(out);
    }
}

#[test]
fn decode_range_split_mode_multiple() {
    let wire = encode_one(&encode_meta_f32(4), &f32_bytes(&[10.0, 20.0, 30.0, 40.0]));
    let offsets = [0u64, 2u64];
    let counts = [1u64, 2u64];
    let mut out = [
        TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        },
        TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        },
    ];
    let mut out_count = 0usize;
    let err = tgm_decode_range(
        wire.as_ptr(),
        wire.len(),
        0,
        offsets.as_ptr(),
        counts.as_ptr(),
        2,
        1,
        0,
        0, // split
        out.as_mut_ptr(),
        &mut out_count,
    );
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    assert_eq!(out_count, 2);
    for b in out {
        if !b.data.is_null() {
            tgm_bytes_free(b);
        }
    }
}

#[test]
fn decode_range_null_buf_is_invalid_arg() {
    let offsets = [0u64];
    let counts = [1u64];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let mut out_count = 0usize;
    let err = tgm_decode_range(
        ptr::null(),
        16,
        0,
        offsets.as_ptr(),
        counts.as_ptr(),
        1,
        1,
        0,
        0,
        &mut out,
        &mut out_count,
    );
    assert!(matches!(err, TgmError::InvalidArg));
}

#[test]
fn decode_range_null_offsets_with_ranges_is_invalid_arg() {
    let wire = encode_one(&encode_meta_f32(4), &f32_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let mut out_count = 0usize;
    let err = tgm_decode_range(
        wire.as_ptr(),
        wire.len(),
        0,
        ptr::null(),
        ptr::null(),
        1,
        1,
        0,
        0,
        &mut out,
        &mut out_count,
    );
    assert!(matches!(err, TgmError::InvalidArg));
}

#[test]
fn decode_range_out_of_range_object_is_error() {
    let wire = encode_one(&encode_meta_f32(4), &f32_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let offsets = [0u64];
    let counts = [1u64];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let mut out_count = 0usize;
    let err = tgm_decode_range(
        wire.as_ptr(),
        wire.len(),
        99,
        offsets.as_ptr(),
        counts.as_ptr(),
        1,
        1,
        0,
        1,
        &mut out,
        &mut out_count,
    );
    assert!(!matches!(err, TgmError::Ok));
    if !out.data.is_null() {
        tgm_bytes_free(out);
    }
}

// ── tgm_scan / scan_count / scan_entry / scan_free ─────────────────────

#[test]
fn scan_over_two_messages() {
    let m0 = encode_one(&encode_meta_f32(2), &f32_bytes(&[1.0, 2.0]));
    let m1 = encode_one(&encode_meta_f32(2), &f32_bytes(&[3.0, 4.0]));
    let mut buf = m0.clone();
    buf.extend_from_slice(&m1);

    let mut result: *mut TgmScanResult = ptr::null_mut();
    let err = tgm_scan(buf.as_ptr(), buf.len(), &mut result);
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    assert_eq!(tgm_scan_count(result), 2);

    let e0 = tgm_scan_entry(result, 0);
    assert_eq!(e0.offset, 0);
    assert_eq!(e0.length, m0.len());
    let e1 = tgm_scan_entry(result, 1);
    assert_eq!(e1.offset, m0.len());

    // Out-of-range entry returns the sentinel.
    let bad = tgm_scan_entry(result, 99);
    assert_eq!(bad.offset, usize::MAX);

    tgm_scan_free(result);
    tgm_scan_free(ptr::null_mut()); // no-op
}

#[test]
fn scan_null_args() {
    let buf = [0u8; 4];
    let err = tgm_scan(buf.as_ptr(), buf.len(), ptr::null_mut());
    assert!(matches!(err, TgmError::InvalidArg));
    let err = tgm_scan(ptr::null(), 4, ptr::null_mut());
    assert!(matches!(err, TgmError::InvalidArg));
    // accessors on a null handle.
    assert_eq!(tgm_scan_count(ptr::null()), 0);
    let s = tgm_scan_entry(ptr::null(), 0);
    assert_eq!(s.offset, usize::MAX);
}

// ── message accessors (all string/array getters + out of range) ────────

#[test]
fn message_accessors_full_coverage() {
    let wire = encode_one(&encode_meta_f32(4), &f32_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let mut msg: *mut TgmMessage = ptr::null_mut();
    assert!(matches!(
        tgm_decode(wire.as_ptr(), wire.len(), 1, 0, 0, &mut msg),
        TgmError::Ok
    ));

    assert_eq!(tgm_message_version(msg) as u16, tensogram::WIRE_VERSION);
    assert_eq!(tgm_message_num_objects(msg), 1);
    assert_eq!(tgm_message_num_decoded(msg), 1);
    assert_eq!(tgm_object_ndim(msg, 0), 1);

    let shape = tgm_object_shape(msg, 0);
    assert!(!shape.is_null());
    assert_eq!(unsafe { *shape }, 4);
    let strides = tgm_object_strides(msg, 0);
    assert!(!strides.is_null());

    let dtype = tgm_object_dtype(msg, 0);
    assert_eq!(
        unsafe { CStr::from_ptr(dtype) }.to_str().unwrap(),
        "float32"
    );
    let ty = tgm_object_type(msg, 0);
    assert!(!ty.is_null());
    let bo = tgm_object_byte_order(msg, 0);
    assert!(!bo.is_null());
    let filter = tgm_object_filter(msg, 0);
    assert_eq!(unsafe { CStr::from_ptr(filter) }.to_str().unwrap(), "none");
    let comp = tgm_object_compression(msg, 0);
    assert_eq!(unsafe { CStr::from_ptr(comp) }.to_str().unwrap(), "none");
    let enc = tgm_payload_encoding(msg, 0);
    assert_eq!(unsafe { CStr::from_ptr(enc) }.to_str().unwrap(), "none");

    // No hash present (encoded without hash) -> 0 / null.
    assert_eq!(tgm_payload_has_hash(msg, 0), 0);
    assert!(tgm_object_hash_type(msg, 0).is_null());
    assert!(tgm_object_hash_value(msg, 0).is_null());

    // Out of range -> null / zero sentinels.
    assert_eq!(tgm_object_ndim(msg, 99), 0);
    assert!(tgm_object_shape(msg, 99).is_null());
    assert!(tgm_object_strides(msg, 99).is_null());
    assert!(tgm_object_dtype(msg, 99).is_null());
    assert!(tgm_object_type(msg, 99).is_null());
    assert!(tgm_object_byte_order(msg, 99).is_null());
    assert!(tgm_object_filter(msg, 99).is_null());
    assert!(tgm_object_compression(msg, 99).is_null());
    assert!(tgm_payload_encoding(msg, 99).is_null());
    assert_eq!(tgm_payload_has_hash(msg, 99), 0);

    let mut olen = 99usize;
    assert!(tgm_object_data(msg, 99, &mut olen).is_null());
    assert_eq!(olen, 0);

    // Valid object with a NULL out_len pointer: returns the data ptr
    // without writing the length.
    let dptr = tgm_object_data(msg, 0, ptr::null_mut());
    assert!(!dptr.is_null());

    tgm_message_free(msg);
}

#[test]
fn message_accessors_on_null_handle() {
    assert_eq!(tgm_message_version(ptr::null()), 0);
    assert_eq!(tgm_message_num_objects(ptr::null()), 0);
    assert_eq!(tgm_message_num_decoded(ptr::null()), 0);
    assert_eq!(tgm_object_ndim(ptr::null(), 0), 0);
    assert!(tgm_object_shape(ptr::null(), 0).is_null());
    assert!(tgm_object_strides(ptr::null(), 0).is_null());
    assert!(tgm_object_dtype(ptr::null(), 0).is_null());
    assert!(tgm_object_type(ptr::null(), 0).is_null());
    assert!(tgm_object_byte_order(ptr::null(), 0).is_null());
    assert!(tgm_object_filter(ptr::null(), 0).is_null());
    assert!(tgm_object_compression(ptr::null(), 0).is_null());
    assert!(tgm_payload_encoding(ptr::null(), 0).is_null());
    assert_eq!(tgm_payload_has_hash(ptr::null(), 0), 0);
    let mut olen = 0usize;
    assert!(tgm_object_data(ptr::null(), 0, &mut olen).is_null());
    // null out_len pointer must not crash.
    assert!(tgm_object_data(ptr::null(), 0, ptr::null_mut()).is_null());
}

#[test]
fn message_accessors_hash_present() {
    let meta = encode_meta_f32(2);
    let data = f32_bytes(&[1.0, 2.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    let algo = cstring("xxh3");
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    assert!(matches!(
        tgm_encode(
            meta.as_ptr(),
            ptrs.as_ptr(),
            lens.as_ptr(),
            1,
            algo.as_ptr(),
            0,
            &mut out,
        ),
        TgmError::Ok
    ));
    let wire = unsafe { std::slice::from_raw_parts(out.data, out.len) }.to_vec();
    tgm_bytes_free(out);

    let mut msg: *mut TgmMessage = ptr::null_mut();
    assert!(matches!(
        tgm_decode(wire.as_ptr(), wire.len(), 1, 0, 0, &mut msg),
        TgmError::Ok
    ));
    assert_eq!(tgm_payload_has_hash(msg, 0), 1);
    let ht = tgm_object_hash_type(msg, 0);
    assert!(!ht.is_null());
    assert_eq!(unsafe { CStr::from_ptr(ht) }.to_str().unwrap(), "xxh3");
    let hv = tgm_object_hash_value(msg, 0);
    assert!(!hv.is_null());
    assert!(!unsafe { CStr::from_ptr(hv) }.to_str().unwrap().is_empty());
    tgm_message_free(msg);
}

// ── tgm_message_metadata ───────────────────────────────────────────────

#[test]
fn message_metadata_extract() {
    let wire = encode_one(&encode_meta_f32(2), &f32_bytes(&[1.0, 2.0]));
    let mut msg: *mut TgmMessage = ptr::null_mut();
    assert!(matches!(
        tgm_decode(wire.as_ptr(), wire.len(), 1, 0, 0, &mut msg),
        TgmError::Ok
    ));
    let mut meta: *mut TgmMetadata = ptr::null_mut();
    assert!(matches!(tgm_message_metadata(msg, &mut meta), TgmError::Ok));
    assert!(!meta.is_null());
    assert_eq!(tgm_metadata_version(meta) as u16, tensogram::WIRE_VERSION);
    tgm_metadata_free(meta);
    tgm_message_free(msg);
}

#[test]
fn message_metadata_null_args_is_invalid_arg() {
    let mut meta: *mut TgmMetadata = ptr::null_mut();
    assert!(matches!(
        tgm_message_metadata(ptr::null(), &mut meta),
        TgmError::InvalidArg
    ));
    let wire = encode_one(&encode_meta_f32(1), &f32_bytes(&[1.0]));
    let mut msg: *mut TgmMessage = ptr::null_mut();
    assert!(matches!(
        tgm_decode(wire.as_ptr(), wire.len(), 1, 0, 0, &mut msg),
        TgmError::Ok
    ));
    assert!(matches!(
        tgm_message_metadata(msg, ptr::null_mut()),
        TgmError::InvalidArg
    ));
    tgm_message_free(msg);
}

// ── metadata accessors (string getter, version on null) ────────────────

#[test]
fn metadata_get_string_roundtrip() {
    let meta_json = cstring(&format!(
        concat!(
            r#"{{"descriptors":[{{"type":"ntensor","ndim":1,"shape":[2],"#,
            r#""strides":[1],"dtype":"float32","byte_order":"{bo}","#,
            r#""encoding":"none","filter":"none","compression":"none"}}],"#,
            r#""base":[{{"centre":"ecmwf"}}]}}"#
        ),
        bo = native_bo()
    ));
    let wire = encode_one(&meta_json, &f32_bytes(&[1.0, 2.0]));
    let mut meta: *mut TgmMetadata = ptr::null_mut();
    assert!(matches!(
        tgm_decode_metadata(wire.as_ptr(), wire.len(), &mut meta),
        TgmError::Ok
    ));
    let key = cstring("centre");
    let v = tgm_metadata_get_string(meta, key.as_ptr());
    assert!(!v.is_null());
    assert_eq!(unsafe { CStr::from_ptr(v) }.to_str().unwrap(), "ecmwf");
    // version pseudo-key.
    let vk = cstring("version");
    let vv = tgm_metadata_get_string(meta, vk.as_ptr());
    assert!(!vv.is_null());
    // missing key -> null.
    let mk = cstring("nope");
    assert!(tgm_metadata_get_string(meta, mk.as_ptr()).is_null());
    assert_eq!(tgm_metadata_num_objects(meta), 1);
    tgm_metadata_free(meta);
}

#[test]
fn metadata_accessors_on_null() {
    assert_eq!(tgm_metadata_version(ptr::null()), 0);
    assert_eq!(tgm_metadata_num_objects(ptr::null()), 0);
    let key = cstring("anything");
    assert!(tgm_metadata_get_string(ptr::null(), key.as_ptr()).is_null());
}

// ── object iterator ────────────────────────────────────────────────────

#[test]
fn object_iter_over_message() {
    // Two objects in one message.
    let meta = cstring(&format!(
        concat!(r#"{{"descriptors":[{d},{d}]}}"#,),
        d = format!(
            concat!(
                r#"{{"type":"ntensor","ndim":1,"shape":[2],"strides":[1],"#,
                r#""dtype":"float32","byte_order":"{bo}","encoding":"none","#,
                r#""filter":"none","compression":"none"}}"#
            ),
            bo = native_bo()
        )
    ));
    let d0 = f32_bytes(&[1.0, 2.0]);
    let d1 = f32_bytes(&[3.0, 4.0]);
    let ptrs = [d0.as_ptr(), d1.as_ptr()];
    let lens = [d0.len(), d1.len()];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    assert!(matches!(
        tgm_encode(
            meta.as_ptr(),
            ptrs.as_ptr(),
            lens.as_ptr(),
            2,
            ptr::null(),
            0,
            &mut out,
        ),
        TgmError::Ok
    ));
    let wire = unsafe { std::slice::from_raw_parts(out.data, out.len) }.to_vec();
    tgm_bytes_free(out);

    let mut iter: *mut TgmObjectIter = ptr::null_mut();
    let err = tgm_object_iter_create(wire.as_ptr(), wire.len(), 1, 0, &mut iter);
    assert!(matches!(err, TgmError::Ok), "{err:?}");

    let mut seen = 0;
    loop {
        let mut m: *mut TgmMessage = ptr::null_mut();
        match tgm_object_iter_next(iter, &mut m) {
            TgmError::Ok => {
                assert_eq!(tgm_message_num_objects(m), 1);
                tgm_message_free(m);
                seen += 1;
            }
            TgmError::EndOfIter => break,
            other => panic!("object_iter_next: {other:?}"),
        }
    }
    assert_eq!(seen, 2);
    tgm_object_iter_free(iter);
    tgm_object_iter_free(ptr::null_mut()); // no-op
}

#[test]
fn object_iter_null_args() {
    let mut iter: *mut TgmObjectIter = ptr::null_mut();
    assert!(matches!(
        tgm_object_iter_create(ptr::null(), 16, 1, 0, &mut iter),
        TgmError::InvalidArg
    ));
    let buf = [0u8; 4];
    assert!(matches!(
        tgm_object_iter_create(buf.as_ptr(), buf.len(), 1, 0, ptr::null_mut()),
        TgmError::InvalidArg
    ));
    let mut m: *mut TgmMessage = ptr::null_mut();
    assert!(matches!(
        tgm_object_iter_next(ptr::null_mut(), &mut m),
        TgmError::InvalidArg
    ));
}

#[test]
fn object_iter_create_garbage_is_error() {
    let g = [0xde_u8, 0xad, 0xbe, 0xef, 0, 0, 0, 0];
    let mut iter: *mut TgmObjectIter = ptr::null_mut();
    let err = tgm_object_iter_create(g.as_ptr(), g.len(), 1, 0, &mut iter);
    assert!(!matches!(err, TgmError::Ok));
}

// ── tgm_error_string ───────────────────────────────────────────────────

#[test]
fn error_string_all_codes() {
    let cases = [
        (TgmError::Ok, "ok"),
        (TgmError::Framing, "framing error"),
        (TgmError::Metadata, "metadata error"),
        (TgmError::Encoding, "encoding error"),
        (TgmError::Compression, "compression error"),
        (TgmError::Object, "object error"),
        (TgmError::Io, "I/O error"),
        (TgmError::HashMismatch, "hash mismatch"),
        (TgmError::InvalidArg, "invalid argument"),
        (TgmError::EndOfIter, "end of iteration"),
        (TgmError::Remote, "remote error"),
        (TgmError::MissingHash, "missing hash"),
        (TgmError::Timeout, "async task timed out"),
        (TgmError::Cancelled, "async task cancelled"),
    ];
    for (code, expected) in cases {
        let p = tgm_error_string(code);
        assert!(!p.is_null());
        assert_eq!(unsafe { CStr::from_ptr(p) }.to_str().unwrap(), expected);
    }
}

// ── tgm_last_error ─────────────────────────────────────────────────────

#[test]
fn last_error_populated_after_failure() {
    // Trigger an error and confirm tgm_last_error returns a message.
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let _ = tgm_encode(
        ptr::null(),
        ptr::null(),
        ptr::null(),
        0,
        ptr::null(),
        0,
        &mut out,
    );
    let p = tgm_last_error();
    assert!(!p.is_null());
    let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap();
    assert!(s.contains("null"), "got: {s}");
}

// ── tgm_simple_packing_compute_params ──────────────────────────────────

#[test]
fn simple_packing_compute_params_ok() {
    let vals = [1.0f64, 2.0, 3.0, 4.0];
    let mut refv = 0.0f64;
    let mut bsf = 0i32;
    let err =
        tgm_simple_packing_compute_params(vals.as_ptr(), vals.len(), 16, 0, &mut refv, &mut bsf);
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    assert!(refv <= 1.0);
}

#[test]
fn simple_packing_null_args_is_invalid_arg() {
    let vals = [1.0f64];
    let mut refv = 0.0f64;
    let mut bsf = 0i32;
    assert!(matches!(
        tgm_simple_packing_compute_params(ptr::null(), 1, 16, 0, &mut refv, &mut bsf),
        TgmError::InvalidArg
    ));
    assert!(matches!(
        tgm_simple_packing_compute_params(vals.as_ptr(), 1, 16, 0, ptr::null_mut(), &mut bsf),
        TgmError::InvalidArg
    ));
    assert!(matches!(
        tgm_simple_packing_compute_params(vals.as_ptr(), 1, 16, 0, &mut refv, ptr::null_mut()),
        TgmError::InvalidArg
    ));
}

#[test]
fn simple_packing_nan_is_encoding_error() {
    let vals = [f64::NAN, 1.0];
    let mut refv = 0.0f64;
    let mut bsf = 0i32;
    let err =
        tgm_simple_packing_compute_params(vals.as_ptr(), vals.len(), 16, 0, &mut refv, &mut bsf);
    assert!(matches!(err, TgmError::Encoding), "{err:?}");
}

// ── tgm_compute_hash ───────────────────────────────────────────────────

#[test]
fn compute_hash_ok_default_and_xxh3() {
    let data = [1u8, 2, 3, 4];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_compute_hash(data.as_ptr(), data.len(), ptr::null(), &mut out);
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    assert!(out.len > 0);
    tgm_bytes_free(out);

    let algo = cstring("xxh3");
    let mut out2 = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    assert!(matches!(
        tgm_compute_hash(data.as_ptr(), data.len(), algo.as_ptr(), &mut out2),
        TgmError::Ok
    ));
    assert!(out2.len > 0);
    tgm_bytes_free(out2);
}

#[test]
fn compute_hash_null_args_is_invalid_arg() {
    let data = [1u8, 2, 3];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    assert!(matches!(
        tgm_compute_hash(ptr::null(), 0, ptr::null(), &mut out),
        TgmError::InvalidArg
    ));
    assert!(matches!(
        tgm_compute_hash(data.as_ptr(), data.len(), ptr::null(), ptr::null_mut()),
        TgmError::InvalidArg
    ));
}

#[test]
fn compute_hash_bad_algo_is_invalid_arg() {
    let data = [1u8, 2, 3];
    let algo = cstring("sha256");
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    assert!(matches!(
        tgm_compute_hash(data.as_ptr(), data.len(), algo.as_ptr(), &mut out),
        TgmError::InvalidArg
    ));
}

#[test]
fn compute_hash_invalid_utf8_algo_is_invalid_arg() {
    let data = [1u8, 2, 3];
    let bad = [0xff_u8, 0xfe, 0x00];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    assert!(matches!(
        tgm_compute_hash(
            data.as_ptr(),
            data.len(),
            bad.as_ptr() as *const std::os::raw::c_char,
            &mut out
        ),
        TgmError::InvalidArg
    ));
}

// ── tgm_doctor_to_json ─────────────────────────────────────────────────

#[test]
fn doctor_to_json_ok() {
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_doctor_to_json(&mut out);
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    assert!(out.len > 0);
    let json = unsafe { std::slice::from_raw_parts(out.data, out.len) };
    assert!(serde_json::from_slice::<serde_json::Value>(json).is_ok());
    tgm_bytes_free(out);
}

#[test]
fn doctor_to_json_null_out_is_invalid_arg() {
    assert!(matches!(
        tgm_doctor_to_json(ptr::null_mut()),
        TgmError::InvalidArg
    ));
}

// ── streaming encoder: preceder / pre_encoded / write error paths ──────

#[test]
fn streaming_encoder_preceder_and_pre_encoded() {
    let dir = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join("tmp");
    std::fs::create_dir_all(&dir).unwrap();
    let path_buf = dir.join(format!("tgm_stream_{}.tgm", std::process::id()));
    let path = cstring(path_buf.to_str().unwrap());
    let meta = cstring("{}");
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_streaming_encoder_create(path.as_ptr(), meta.as_ptr(), ptr::null(), 0, &mut enc),
        TgmError::Ok
    ));

    // write_preceder then a normal write.
    let preceder = cstring(r#"{"units":"K"}"#);
    assert!(matches!(
        tgm_streaming_encoder_write_preceder(enc, preceder.as_ptr()),
        TgmError::Ok
    ));
    let desc = stream_desc_f32(4);
    let d = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    assert!(matches!(
        tgm_streaming_encoder_write(enc, desc.as_ptr(), d.as_ptr(), d.len()),
        TgmError::Ok
    ));

    // pre-encoded write (encoding "none" -> bytes pass through).
    let d2 = f32_bytes(&[5.0, 6.0, 7.0, 8.0]);
    assert!(matches!(
        tgm_streaming_encoder_write_pre_encoded(enc, desc.as_ptr(), d2.as_ptr(), d2.len()),
        TgmError::Ok
    ));

    assert_eq!(tgm_streaming_encoder_count(enc), 2);
    assert!(matches!(tgm_streaming_encoder_finish(enc), TgmError::Ok));
    // Double finish -> already finished.
    assert!(matches!(
        tgm_streaming_encoder_finish(enc),
        TgmError::InvalidArg
    ));
    tgm_streaming_encoder_free(enc);
    let _ = std::fs::remove_file(&path_buf);
}

#[test]
fn streaming_encoder_count_on_null_is_zero() {
    assert_eq!(tgm_streaming_encoder_count(ptr::null()), 0);
}

#[test]
fn streaming_encoder_finish_null_is_invalid_arg() {
    assert!(matches!(
        tgm_streaming_encoder_finish(ptr::null_mut()),
        TgmError::InvalidArg
    ));
}

#[test]
fn streaming_encoder_write_null_args_is_invalid_arg() {
    assert!(matches!(
        tgm_streaming_encoder_write(ptr::null_mut(), ptr::null(), ptr::null(), 0),
        TgmError::InvalidArg
    ));
    assert!(matches!(
        tgm_streaming_encoder_write_pre_encoded(ptr::null_mut(), ptr::null(), ptr::null(), 0),
        TgmError::InvalidArg
    ));
    assert!(matches!(
        tgm_streaming_encoder_write_preceder(ptr::null_mut(), ptr::null()),
        TgmError::InvalidArg
    ));
}

#[test]
fn streaming_encoder_write_after_finish_is_invalid_arg() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let meta = cstring("{}");
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_streaming_encoder_create(path.as_ptr(), meta.as_ptr(), ptr::null(), 0, &mut enc),
        TgmError::Ok
    ));
    assert!(matches!(tgm_streaming_encoder_finish(enc), TgmError::Ok));
    // Now the inner encoder is consumed; writes must fail.
    let desc = stream_desc_f32(2);
    let d = f32_bytes(&[1.0, 2.0]);
    assert!(matches!(
        tgm_streaming_encoder_write(enc, desc.as_ptr(), d.as_ptr(), d.len()),
        TgmError::InvalidArg
    ));
    assert!(matches!(
        tgm_streaming_encoder_write_pre_encoded(enc, desc.as_ptr(), d.as_ptr(), d.len()),
        TgmError::InvalidArg
    ));
    let preceder = cstring(r#"{"x":1}"#);
    assert!(matches!(
        tgm_streaming_encoder_write_preceder(enc, preceder.as_ptr()),
        TgmError::InvalidArg
    ));
    tgm_streaming_encoder_free(enc);
}

#[test]
fn streaming_encoder_write_bad_descriptor_is_metadata_error() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let meta = cstring("{}");
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_streaming_encoder_create(path.as_ptr(), meta.as_ptr(), ptr::null(), 0, &mut enc),
        TgmError::Ok
    ));
    let bad_desc = cstring("{ not json ");
    let d = f32_bytes(&[1.0]);
    assert!(matches!(
        tgm_streaming_encoder_write(enc, bad_desc.as_ptr(), d.as_ptr(), d.len()),
        TgmError::Metadata
    ));
    assert!(matches!(
        tgm_streaming_encoder_write_pre_encoded(enc, bad_desc.as_ptr(), d.as_ptr(), d.len()),
        TgmError::Metadata
    ));
    tgm_streaming_encoder_free(enc);
}

#[test]
fn streaming_encoder_preceder_non_object_is_metadata_error() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let meta = cstring("{}");
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_streaming_encoder_create(path.as_ptr(), meta.as_ptr(), ptr::null(), 0, &mut enc),
        TgmError::Ok
    ));
    // A JSON array is valid JSON but not an object.
    let arr = cstring("[1,2,3]");
    assert!(matches!(
        tgm_streaming_encoder_write_preceder(enc, arr.as_ptr()),
        TgmError::Metadata
    ));
    let bad = cstring("{ not json");
    assert!(matches!(
        tgm_streaming_encoder_write_preceder(enc, bad.as_ptr()),
        TgmError::Metadata
    ));
    tgm_streaming_encoder_free(enc);
}

#[test]
fn streaming_encoder_create_invalid_utf8_path_is_invalid_arg() {
    let bad = [0xff_u8, 0xfe, 0x00];
    let meta = cstring("{}");
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_streaming_encoder_create(
            bad.as_ptr() as *const std::os::raw::c_char,
            meta.as_ptr(),
            ptr::null(),
            0,
            &mut enc
        ),
        TgmError::InvalidArg
    ));
}

#[test]
fn streaming_encoder_create_bad_hash_algo_is_invalid_arg() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let meta = cstring("{}");
    let algo = cstring("sha256");
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_streaming_encoder_create(path.as_ptr(), meta.as_ptr(), algo.as_ptr(), 0, &mut enc),
        TgmError::InvalidArg
    ));
}

// ── streaming encoder with options ─────────────────────────────────────

#[test]
fn streaming_encoder_create_with_options_ok() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let meta = cstring("{}");
    let opts = TgmEncodeMaskOptions {
        allow_nan: true,
        allow_inf: true,
        nan_mask_method: ptr::null(),
        pos_inf_mask_method: ptr::null(),
        neg_inf_mask_method: ptr::null(),
        small_mask_threshold_bytes: 0,
    };
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    let err = tgm_streaming_encoder_create_with_options(
        path.as_ptr(),
        meta.as_ptr(),
        ptr::null(),
        0,
        &opts,
        &mut enc,
    );
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    assert!(!enc.is_null());
    // Write a NaN object: allowed because allow_nan = true.
    let desc = stream_desc_f32(2);
    let d = f32_bytes(&[f32::NAN, 1.0]);
    assert!(matches!(
        tgm_streaming_encoder_write(enc, desc.as_ptr(), d.as_ptr(), d.len()),
        TgmError::Ok
    ));
    assert!(matches!(tgm_streaming_encoder_finish(enc), TgmError::Ok));
    tgm_streaming_encoder_free(enc);
}

#[test]
fn streaming_encoder_create_with_options_null_args_is_invalid_arg() {
    let meta = cstring("{}");
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_streaming_encoder_create_with_options(
            ptr::null(),
            meta.as_ptr(),
            ptr::null(),
            0,
            ptr::null(),
            &mut enc
        ),
        TgmError::InvalidArg
    ));
}

#[test]
fn streaming_encoder_create_with_options_invalid_utf8_path() {
    let bad = [0xff_u8, 0xfe, 0x00];
    let meta = cstring("{}");
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_streaming_encoder_create_with_options(
            bad.as_ptr() as *const std::os::raw::c_char,
            meta.as_ptr(),
            ptr::null(),
            0,
            ptr::null(),
            &mut enc
        ),
        TgmError::InvalidArg
    ));
}

#[test]
fn streaming_encoder_create_with_options_invalid_utf8_metadata() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let bad = [0xff_u8, 0xfe, 0x00];
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_streaming_encoder_create_with_options(
            path.as_ptr(),
            bad.as_ptr() as *const std::os::raw::c_char,
            ptr::null(),
            0,
            ptr::null(),
            &mut enc
        ),
        TgmError::InvalidArg
    ));
}

#[test]
fn streaming_encoder_create_with_options_bad_metadata_is_metadata_error() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let bad = cstring("{ not json ");
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_streaming_encoder_create_with_options(
            path.as_ptr(),
            bad.as_ptr(),
            ptr::null(),
            0,
            ptr::null(),
            &mut enc
        ),
        TgmError::Metadata
    ));
}

#[test]
fn streaming_encoder_create_with_options_bad_hash_algo() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let meta = cstring("{}");
    let algo = cstring("sha256");
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_streaming_encoder_create_with_options(
            path.as_ptr(),
            meta.as_ptr(),
            algo.as_ptr(),
            0,
            ptr::null(),
            &mut enc
        ),
        TgmError::InvalidArg
    ));
}

#[test]
fn streaming_encoder_create_with_options_io_error() {
    let path = cstring("/nonexistent/tensogram/dir/out.tgm");
    let meta = cstring("{}");
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_streaming_encoder_create_with_options(
            path.as_ptr(),
            meta.as_ptr(),
            ptr::null(),
            0,
            ptr::null(),
            &mut enc
        ),
        TgmError::Io
    ));
}

#[test]
fn streaming_encoder_create_with_options_bad_mask_method() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let meta = cstring("{}");
    let bad_method = cstring("nope-codec");
    let opts = TgmEncodeMaskOptions {
        allow_nan: true,
        allow_inf: false,
        nan_mask_method: bad_method.as_ptr(),
        pos_inf_mask_method: ptr::null(),
        neg_inf_mask_method: ptr::null(),
        small_mask_threshold_bytes: 0,
    };
    let mut enc: *mut TgmStreamingEncoder = ptr::null_mut();
    assert!(matches!(
        tgm_streaming_encoder_create_with_options(
            path.as_ptr(),
            meta.as_ptr(),
            ptr::null(),
            0,
            &opts,
            &mut enc
        ),
        TgmError::InvalidArg
    ));
}

// ── tgm_validate: null buf with len, default level ─────────────────────

#[test]
fn validate_null_buf_zero_len_is_ok() {
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    // NULL buf + len 0 + NULL level (=> "default") is a valid empty-buffer call.
    let err = tgm_validate(ptr::null(), 0, ptr::null(), 0, &mut out);
    assert!(matches!(err, TgmError::Ok), "{err:?}");
    assert!(out.len > 0);
    tgm_bytes_free(out);
}

#[test]
fn validate_null_buf_nonzero_len_is_invalid_arg() {
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_validate(ptr::null(), 16, ptr::null(), 0, &mut out);
    assert!(matches!(err, TgmError::InvalidArg));
}

#[test]
fn validate_invalid_utf8_level_is_invalid_arg() {
    let wire = encode_one(&encode_meta_f32(1), &f32_bytes(&[1.0]));
    let bad = [0xff_u8, 0xfe, 0x00];
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_validate(
        wire.as_ptr(),
        wire.len(),
        bad.as_ptr() as *const std::os::raw::c_char,
        0,
        &mut out,
    );
    assert!(matches!(err, TgmError::InvalidArg));
}

#[test]
fn validate_all_levels() {
    let wire = encode_one(&encode_meta_f32(2), &f32_bytes(&[1.0, 2.0]));
    for lvl in ["quick", "default", "checksum", "full"] {
        let level = cstring(lvl);
        let mut out = TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = tgm_validate(wire.as_ptr(), wire.len(), level.as_ptr(), 1, &mut out);
        assert!(matches!(err, TgmError::Ok), "{lvl}: {err:?}");
        assert!(out.len > 0);
        tgm_bytes_free(out);
    }
}

#[test]
fn validate_file_invalid_utf8_path_is_invalid_arg() {
    let bad = [0xff_u8, 0xfe, 0x00];
    let level = cstring("quick");
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_validate_file(
        bad.as_ptr() as *const std::os::raw::c_char,
        level.as_ptr(),
        0,
        &mut out,
    );
    assert!(matches!(err, TgmError::InvalidArg));
}

#[test]
fn validate_file_invalid_level_is_invalid_arg() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let level = cstring("bogus");
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    let err = tgm_validate_file(path.as_ptr(), level.as_ptr(), 0, &mut out);
    assert!(matches!(err, TgmError::InvalidArg));
}
