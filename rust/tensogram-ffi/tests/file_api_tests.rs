// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

//! Integration tests for the synchronous FFI file API, the buffer / file
//! iterators, and the metadata getters, through the C ABI (happy + error
//! paths). These entry points are exercised by the C/Python/Fortran suites
//! but never by `cargo llvm-cov`.

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

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(vals.len() * 4);
    for &x in vals {
        v.extend_from_slice(&x.to_ne_bytes());
    }
    v
}

/// `{"descriptors":[<float32 1-D>], <extra>}` — `extra` is a raw top-level
/// JSON fragment (e.g. `"base":[{...}]`) or empty.
fn encode_meta(n: usize, extra: &str) -> CString {
    let comma = if extra.is_empty() { "" } else { "," };
    cstring(&format!(
        concat!(
            r#"{{"descriptors":[{{"type":"ntensor","ndim":1,"shape":[{n}],"#,
            r#""strides":[1],"dtype":"float32","byte_order":"{bo}","#,
            r#""encoding":"none","filter":"none","compression":"none"}}]{c}{extra}}}"#
        ),
        n = n,
        bo = native_bo(),
        c = comma,
        extra = extra
    ))
}

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

/// Append one float32 object to an open file via `tgm_file_append`.
fn file_append_one(file: *mut TgmFile, meta: &CString, data: &[u8]) -> TgmError {
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    tgm_file_append(
        file,
        meta.as_ptr(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        1,
        ptr::null(),
        0,
    )
}

// ── file API round-trip ────────────────────────────────────────────────

#[test]
fn file_create_append_count_decode_read_path_close() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());

    let mut file: *mut TgmFile = ptr::null_mut();
    let err = tgm_file_create(path.as_ptr(), &mut file);
    assert!(matches!(err, TgmError::Ok), "create: {err:?}");
    assert!(!file.is_null());

    let meta = encode_meta(4, "");
    assert!(matches!(
        file_append_one(file, &meta, &f32_bytes(&[1.0, 2.0, 3.0, 4.0])),
        TgmError::Ok
    ));
    assert!(matches!(
        file_append_one(file, &meta, &f32_bytes(&[5.0, 6.0, 7.0, 8.0])),
        TgmError::Ok
    ));

    let mut count = 0usize;
    assert!(matches!(
        tgm_file_message_count(file, &mut count),
        TgmError::Ok
    ));
    assert_eq!(count, 2);

    // decode_message(1)
    let mut msg: *mut TgmMessage = ptr::null_mut();
    assert!(matches!(
        tgm_file_decode_message(file, 1, 1, 0, 0, &mut msg),
        TgmError::Ok
    ));
    assert_eq!(tgm_message_num_objects(msg), 1);
    let mut olen = 0usize;
    let dptr = tgm_object_data(msg, 0, &mut olen);
    assert_eq!(
        unsafe { std::slice::from_raw_parts(dptr, olen) },
        &f32_bytes(&[5.0, 6.0, 7.0, 8.0])[..]
    );
    tgm_message_free(msg);

    // read_message(0) — raw bytes
    let mut raw = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    assert!(matches!(
        tgm_file_read_message(file, 0, &mut raw),
        TgmError::Ok
    ));
    assert!(raw.len > 0);
    tgm_bytes_free(raw);

    // path
    let p = tgm_file_path(file);
    assert!(!p.is_null());
    let p = unsafe { CStr::from_ptr(p) }.to_str().unwrap();
    assert_eq!(p, f.path().to_str().unwrap());

    tgm_file_close(file);
    tgm_file_close(ptr::null_mut()); // no-op
}

#[test]
fn file_open_and_iterate() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let meta = encode_meta(2, "");

    // Write two messages.
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(matches!(
        tgm_file_create(path.as_ptr(), &mut file),
        TgmError::Ok
    ));
    assert!(matches!(
        file_append_one(file, &meta, &f32_bytes(&[1.0, 2.0])),
        TgmError::Ok
    ));
    assert!(matches!(
        file_append_one(file, &meta, &f32_bytes(&[3.0, 4.0])),
        TgmError::Ok
    ));
    tgm_file_close(file);

    // Reopen and iterate.
    let mut rfile: *mut TgmFile = ptr::null_mut();
    assert!(matches!(
        tgm_file_open(path.as_ptr(), &mut rfile),
        TgmError::Ok
    ));
    let mut iter: *mut TgmFileIter = ptr::null_mut();
    assert!(matches!(
        tgm_file_iter_create(rfile, &mut iter),
        TgmError::Ok
    ));

    let mut seen = 0;
    loop {
        let mut b = TgmBytes {
            data: ptr::null_mut(),
            len: 0,
        };
        let err = tgm_file_iter_next(iter, &mut b);
        match err {
            TgmError::Ok => {
                assert!(b.len > 0);
                tgm_bytes_free(b);
                seen += 1;
            }
            TgmError::EndOfIter => break,
            other => panic!("file_iter_next: {other:?}"),
        }
    }
    assert_eq!(seen, 2);
    tgm_file_iter_free(iter);
    tgm_file_iter_free(ptr::null_mut()); // no-op
    tgm_file_close(rfile);
}

#[test]
fn file_append_raw_roundtrip() {
    // Encode a standalone message, then append it raw to a file.
    let wire = encode_one(&encode_meta(3, ""), &f32_bytes(&[1.0, 2.0, 3.0]));
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(matches!(
        tgm_file_create(path.as_ptr(), &mut file),
        TgmError::Ok
    ));
    assert!(matches!(
        tgm_file_append_raw(file, wire.as_ptr(), wire.len()),
        TgmError::Ok
    ));
    let mut count = 0usize;
    assert!(matches!(
        tgm_file_message_count(file, &mut count),
        TgmError::Ok
    ));
    assert_eq!(count, 1);
    tgm_file_close(file);
}

// ── file API error paths ───────────────────────────────────────────────

#[test]
fn file_create_null_path_is_invalid_arg() {
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(matches!(
        tgm_file_create(ptr::null(), &mut file),
        TgmError::InvalidArg
    ));
}

#[test]
fn file_open_missing_is_error() {
    let path = cstring("/nonexistent/tensogram/missing.tgm");
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(!matches!(
        tgm_file_open(path.as_ptr(), &mut file),
        TgmError::Ok
    ));
}

#[test]
fn file_decode_message_out_of_range_is_error() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(matches!(
        tgm_file_create(path.as_ptr(), &mut file),
        TgmError::Ok
    ));
    assert!(matches!(
        file_append_one(file, &encode_meta(1, ""), &f32_bytes(&[1.0])),
        TgmError::Ok
    ));
    let mut msg: *mut TgmMessage = ptr::null_mut();
    let err = tgm_file_decode_message(file, 99, 1, 0, 0, &mut msg);
    assert!(!matches!(err, TgmError::Ok));
    tgm_file_close(file);
}

#[test]
fn file_message_count_null_out_is_invalid_arg() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(matches!(
        tgm_file_create(path.as_ptr(), &mut file),
        TgmError::Ok
    ));
    assert!(matches!(
        tgm_file_message_count(file, ptr::null_mut()),
        TgmError::InvalidArg
    ));
    tgm_file_close(file);
}

#[test]
fn file_path_on_null_is_null() {
    assert!(tgm_file_path(ptr::null()).is_null());
}

// ── buffer iterator ────────────────────────────────────────────────────

#[test]
fn buffer_iter_over_two_messages() {
    let m0 = encode_one(&encode_meta(2, ""), &f32_bytes(&[1.0, 2.0]));
    let m1 = encode_one(&encode_meta(2, ""), &f32_bytes(&[3.0, 4.0]));
    let mut buf = m0.clone();
    buf.extend_from_slice(&m1);

    let mut iter: *mut TgmBufferIter = ptr::null_mut();
    assert!(matches!(
        tgm_buffer_iter_create(buf.as_ptr(), buf.len(), &mut iter),
        TgmError::Ok
    ));
    assert_eq!(tgm_buffer_iter_count(iter), 2);

    let mut seen = 0;
    loop {
        let mut out_buf: *const u8 = ptr::null();
        let mut out_len = 0usize;
        let err = tgm_buffer_iter_next(iter, &mut out_buf, &mut out_len);
        match err {
            TgmError::Ok => {
                assert!(!out_buf.is_null() && out_len > 0);
                seen += 1;
            }
            TgmError::EndOfIter => break,
            other => panic!("buffer_iter_next: {other:?}"),
        }
    }
    assert_eq!(seen, 2);
    tgm_buffer_iter_free(iter);
    tgm_buffer_iter_free(ptr::null_mut()); // no-op
}

#[test]
fn buffer_iter_null_args_are_invalid() {
    let mut iter: *mut TgmBufferIter = ptr::null_mut();
    assert!(matches!(
        tgm_buffer_iter_create(ptr::null(), 16, &mut iter),
        TgmError::InvalidArg
    ));
    assert_eq!(tgm_buffer_iter_count(ptr::null()), 0);
}

// ── metadata getters ───────────────────────────────────────────────────

#[test]
fn metadata_get_int_and_float() {
    let meta_json = encode_meta(2, r#""base":[{"level":850,"scale":2.5}]"#);
    let wire = encode_one(&meta_json, &f32_bytes(&[1.0, 2.0]));

    let mut meta: *mut TgmMetadata = ptr::null_mut();
    assert!(matches!(
        tgm_decode_metadata(wire.as_ptr(), wire.len(), &mut meta),
        TgmError::Ok
    ));
    assert!(!meta.is_null());

    let k_level = cstring("level");
    let k_scale = cstring("scale");
    let k_missing = cstring("nope");

    assert_eq!(tgm_metadata_get_int(meta, k_level.as_ptr(), -1), 850);
    assert!((tgm_metadata_get_float(meta, k_scale.as_ptr(), -1.0) - 2.5).abs() < 1e-9);
    // Missing keys fall back to the supplied default.
    assert_eq!(tgm_metadata_get_int(meta, k_missing.as_ptr(), 42), 42);
    assert!((tgm_metadata_get_float(meta, k_missing.as_ptr(), 3.5) - 3.5).abs() < 1e-9);

    tgm_metadata_free(meta);
}

#[test]
fn metadata_getters_on_null_return_default() {
    let key = cstring("anything");
    assert_eq!(tgm_metadata_get_int(ptr::null(), key.as_ptr(), 7), 7);
    assert!((tgm_metadata_get_float(ptr::null(), key.as_ptr(), 1.25) - 1.25).abs() < 1e-9);
}

// ── tgm_file_append_with_options ───────────────────────────────────────

fn file_append_with_options(
    file: *mut TgmFile,
    meta: &CString,
    data: &[u8],
    opts: *const TgmEncodeMaskOptions,
) -> TgmError {
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    tgm_file_append_with_options(
        file,
        meta.as_ptr(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        1,
        ptr::null(),
        0,
        opts,
    )
}

#[test]
fn file_append_with_options_null_is_default() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(matches!(
        tgm_file_create(path.as_ptr(), &mut file),
        TgmError::Ok
    ));
    let meta = encode_meta(3, "");
    assert!(matches!(
        file_append_with_options(file, &meta, &f32_bytes(&[1.0, 2.0, 3.0]), ptr::null()),
        TgmError::Ok
    ));
    let mut count = 0usize;
    assert!(matches!(
        tgm_file_message_count(file, &mut count),
        TgmError::Ok
    ));
    assert_eq!(count, 1);
    tgm_file_close(file);
}

#[test]
fn file_append_with_options_allow_nan() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(matches!(
        tgm_file_create(path.as_ptr(), &mut file),
        TgmError::Ok
    ));
    let opts = TgmEncodeMaskOptions {
        allow_nan: true,
        allow_inf: false,
        nan_mask_method: ptr::null(),
        pos_inf_mask_method: ptr::null(),
        neg_inf_mask_method: ptr::null(),
        small_mask_threshold_bytes: 0,
    };
    let meta = encode_meta(2, "");
    assert!(matches!(
        file_append_with_options(file, &meta, &f32_bytes(&[f32::NAN, 1.0]), &opts),
        TgmError::Ok
    ));
    tgm_file_close(file);
}

#[test]
fn file_append_with_options_null_file_is_invalid_arg() {
    let opts = TgmEncodeMaskOptions {
        allow_nan: true,
        allow_inf: false,
        nan_mask_method: ptr::null(),
        pos_inf_mask_method: ptr::null(),
        neg_inf_mask_method: ptr::null(),
        small_mask_threshold_bytes: 0,
    };
    let meta = encode_meta(1, "");
    assert!(matches!(
        file_append_with_options(ptr::null_mut(), &meta, &f32_bytes(&[1.0]), &opts),
        TgmError::InvalidArg
    ));
}

#[test]
fn file_append_with_options_invalid_utf8_metadata_is_invalid_arg() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(matches!(
        tgm_file_create(path.as_ptr(), &mut file),
        TgmError::Ok
    ));
    let bad = [0xff_u8, 0xfe, 0x00];
    let data = f32_bytes(&[1.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    assert!(matches!(
        tgm_file_append_with_options(
            file,
            bad.as_ptr() as *const std::os::raw::c_char,
            ptrs.as_ptr(),
            lens.as_ptr(),
            1,
            ptr::null(),
            0,
            ptr::null(),
        ),
        TgmError::InvalidArg
    ));
    tgm_file_close(file);
}

#[test]
fn file_append_with_options_bad_mask_method_is_invalid_arg() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(matches!(
        tgm_file_create(path.as_ptr(), &mut file),
        TgmError::Ok
    ));
    let bad_method = cstring("nope-codec");
    let opts = TgmEncodeMaskOptions {
        allow_nan: true,
        allow_inf: false,
        nan_mask_method: bad_method.as_ptr(),
        pos_inf_mask_method: ptr::null(),
        neg_inf_mask_method: ptr::null(),
        small_mask_threshold_bytes: 0,
    };
    let meta = encode_meta(2, "");
    assert!(matches!(
        file_append_with_options(file, &meta, &f32_bytes(&[f32::NAN, 1.0]), &opts),
        TgmError::InvalidArg
    ));
    tgm_file_close(file);
}

// ── tgm_file_append error paths ────────────────────────────────────────

#[test]
fn file_append_null_file_is_invalid_arg() {
    let meta = encode_meta(1, "");
    assert!(matches!(
        file_append_one(ptr::null_mut(), &meta, &f32_bytes(&[1.0])),
        TgmError::InvalidArg
    ));
}

#[test]
fn file_append_invalid_utf8_metadata_is_invalid_arg() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(matches!(
        tgm_file_create(path.as_ptr(), &mut file),
        TgmError::Ok
    ));
    let bad = [0xff_u8, 0xfe, 0x00];
    let data = f32_bytes(&[1.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    assert!(matches!(
        tgm_file_append(
            file,
            bad.as_ptr() as *const std::os::raw::c_char,
            ptrs.as_ptr(),
            lens.as_ptr(),
            1,
            ptr::null(),
            0,
        ),
        TgmError::InvalidArg
    ));
    tgm_file_close(file);
}

#[test]
fn file_append_bad_metadata_is_metadata_error() {
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(matches!(
        tgm_file_create(path.as_ptr(), &mut file),
        TgmError::Ok
    ));
    let bad = cstring("{ not json ");
    let data = f32_bytes(&[1.0]);
    let ptrs = [data.as_ptr()];
    let lens = [data.len()];
    assert!(matches!(
        tgm_file_append(
            file,
            bad.as_ptr(),
            ptrs.as_ptr(),
            lens.as_ptr(),
            1,
            ptr::null(),
            0,
        ),
        TgmError::Metadata
    ));
    tgm_file_close(file);
}

// ── tgm_file_open / create error paths ─────────────────────────────────

#[test]
fn file_open_null_out_is_invalid_arg() {
    let path = cstring("/tmp/whatever.tgm");
    assert!(matches!(
        tgm_file_open(path.as_ptr(), ptr::null_mut()),
        TgmError::InvalidArg
    ));
}

#[test]
fn file_open_invalid_utf8_path_is_invalid_arg() {
    let bad = [0xff_u8, 0xfe, 0x00];
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(matches!(
        tgm_file_open(bad.as_ptr() as *const std::os::raw::c_char, &mut file),
        TgmError::InvalidArg
    ));
}

#[test]
fn file_create_invalid_utf8_path_is_invalid_arg() {
    let bad = [0xff_u8, 0xfe, 0x00];
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(matches!(
        tgm_file_create(bad.as_ptr() as *const std::os::raw::c_char, &mut file),
        TgmError::InvalidArg
    ));
}

#[test]
fn file_create_in_missing_dir_is_io_error() {
    // `TensogramFile::create` runs `create_dir_all(parent)` before
    // opening the file, so a merely-absent parent directory is created
    // (and the call succeeds) — especially when the test runs as root in
    // CI, where even `/nonexistent/...` is creatable.  To get a
    // deterministic I/O error regardless of privilege, point the path at
    // a child *under a regular file*: `create_dir_all` then fails with
    // NotADirectory because a path component is not a directory.
    let parent = tempfile::NamedTempFile::new().unwrap();
    let bogus = parent.path().join("sub").join("out.tgm");
    let path = cstring(bogus.to_str().unwrap());
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(!matches!(
        tgm_file_create(path.as_ptr(), &mut file),
        TgmError::Ok
    ));
}

// ── tgm_file_message_count / decode / read null args ───────────────────

#[test]
fn file_message_count_null_file_is_invalid_arg() {
    let mut count = 0usize;
    assert!(matches!(
        tgm_file_message_count(ptr::null_mut(), &mut count),
        TgmError::InvalidArg
    ));
}

#[test]
fn file_decode_message_null_args_is_invalid_arg() {
    let mut msg: *mut TgmMessage = ptr::null_mut();
    assert!(matches!(
        tgm_file_decode_message(ptr::null_mut(), 0, 1, 0, 0, &mut msg),
        TgmError::InvalidArg
    ));
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(matches!(
        tgm_file_create(path.as_ptr(), &mut file),
        TgmError::Ok
    ));
    assert!(matches!(
        tgm_file_decode_message(file, 0, 1, 0, 0, ptr::null_mut()),
        TgmError::InvalidArg
    ));
    tgm_file_close(file);
}

#[test]
fn file_read_message_null_args_and_out_of_range() {
    let mut raw = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    assert!(matches!(
        tgm_file_read_message(ptr::null_mut(), 0, &mut raw),
        TgmError::InvalidArg
    ));

    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(matches!(
        tgm_file_create(path.as_ptr(), &mut file),
        TgmError::Ok
    ));
    assert!(matches!(
        file_append_one(file, &encode_meta(1, ""), &f32_bytes(&[1.0])),
        TgmError::Ok
    ));
    assert!(matches!(
        tgm_file_read_message(file, 0, ptr::null_mut()),
        TgmError::InvalidArg
    ));
    // out-of-range index.
    let err = tgm_file_read_message(file, 99, &mut raw);
    assert!(!matches!(err, TgmError::Ok));
    if !raw.data.is_null() {
        tgm_bytes_free(raw);
    }
    tgm_file_close(file);
}

#[test]
fn file_append_raw_null_args_is_invalid_arg() {
    let buf = [0u8; 4];
    assert!(matches!(
        tgm_file_append_raw(ptr::null_mut(), buf.as_ptr(), buf.len()),
        TgmError::InvalidArg
    ));
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(matches!(
        tgm_file_create(path.as_ptr(), &mut file),
        TgmError::Ok
    ));
    assert!(matches!(
        tgm_file_append_raw(file, ptr::null(), 4),
        TgmError::InvalidArg
    ));
    tgm_file_close(file);
}

// ── tgm_file_iter error paths ──────────────────────────────────────────

#[test]
fn file_iter_create_null_args_is_invalid_arg() {
    let mut iter: *mut TgmFileIter = ptr::null_mut();
    assert!(matches!(
        tgm_file_iter_create(ptr::null_mut(), &mut iter),
        TgmError::InvalidArg
    ));
    let f = tempfile::NamedTempFile::new().unwrap();
    let path = cstring(f.path().to_str().unwrap());
    let mut file: *mut TgmFile = ptr::null_mut();
    assert!(matches!(
        tgm_file_create(path.as_ptr(), &mut file),
        TgmError::Ok
    ));
    assert!(matches!(
        tgm_file_iter_create(file, ptr::null_mut()),
        TgmError::InvalidArg
    ));
    tgm_file_close(file);
}

#[test]
fn file_iter_next_null_args_is_invalid_arg() {
    let mut out = TgmBytes {
        data: ptr::null_mut(),
        len: 0,
    };
    assert!(matches!(
        tgm_file_iter_next(ptr::null_mut(), &mut out),
        TgmError::InvalidArg
    ));
}

// ── tgm_buffer_iter_next null args ─────────────────────────────────────

#[test]
fn buffer_iter_next_null_args_is_invalid_arg() {
    let mut out_buf: *const u8 = ptr::null();
    let mut out_len = 0usize;
    assert!(matches!(
        tgm_buffer_iter_next(ptr::null_mut(), &mut out_buf, &mut out_len),
        TgmError::InvalidArg
    ));
}

// ── metadata getters: invalid-utf8 key returns default ─────────────────

#[test]
fn metadata_getters_invalid_utf8_key_returns_default() {
    let meta_json = encode_meta(2, r#""base":[{"level":850}]"#);
    let wire = encode_one(&meta_json, &f32_bytes(&[1.0, 2.0]));
    let mut meta: *mut TgmMetadata = ptr::null_mut();
    assert!(matches!(
        tgm_decode_metadata(wire.as_ptr(), wire.len(), &mut meta),
        TgmError::Ok
    ));
    let bad = [0xff_u8, 0xfe, 0x00];
    let key = bad.as_ptr() as *const std::os::raw::c_char;
    assert_eq!(tgm_metadata_get_int(meta, key, 5), 5);
    assert!((tgm_metadata_get_float(meta, key, 1.5) - 1.5).abs() < 1e-9);
    assert!(tgm_metadata_get_string(meta, key).is_null());
    // null key.
    assert_eq!(tgm_metadata_get_int(meta, ptr::null(), 9), 9);
    assert!(tgm_metadata_get_string(meta, ptr::null()).is_null());
    tgm_metadata_free(meta);
}
