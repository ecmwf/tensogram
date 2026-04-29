// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Integration tests for the `--allow-nan` / `--allow-inf` CLI flags
//! and matching env vars.  See `docs/src/guide/nan-inf-handling.md`.

use std::process::Command;

use tensogram::{
    ByteOrder, DataObjectDescriptor, Dtype, EncodeOptions, GlobalMetadata, StreamingEncoder,
    encode_pre_encoded,
};

fn cli() -> Command {
    let bin = env!("CARGO_BIN_EXE_tensogram");
    Command::new(bin)
}

fn make_nan_file(path: &std::path::Path) {
    // Build a file with a NaN-bearing float64 payload using
    // encode_pre_encoded (bypasses finite-check).
    use std::collections::BTreeMap;
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![3],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        masks: None,
        params: BTreeMap::new(),
    };
    let data: Vec<u8> = [1.0_f64, f64::NAN, 3.0]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    let meta = GlobalMetadata::default();
    let msg = encode_pre_encoded(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    std::fs::write(path, msg).unwrap();
}

fn make_finite_file(path: &std::path::Path) {
    use std::collections::BTreeMap;
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![4],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        masks: None,
        params: BTreeMap::new(),
    };
    let data: Vec<u8> = [1.0_f64, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    let meta = GlobalMetadata::default();
    let mut buf = Vec::new();
    let enc_opts = EncodeOptions {
        hashing: false,
        ..Default::default()
    };
    let mut enc = StreamingEncoder::new(&mut buf, &meta, &enc_opts).unwrap();
    enc.write_object(&desc, &data).unwrap();
    enc.finish().unwrap();
    std::fs::write(path, &buf).unwrap();
}

// ── Default reject: copy of a NaN-bearing file through reshuffle fails ─────

#[test]
fn reshuffle_without_allow_nan_fails_on_nan_payload() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.tgm");
    let output = dir.path().join("out.tgm");
    make_nan_file(&input);

    let status = cli()
        .args(["reshuffle", "-o"])
        .arg(&output)
        .arg(&input)
        .output()
        .unwrap();
    // reshuffle decodes + re-encodes; NaN trips the default reject policy.
    assert!(
        !status.status.success(),
        "reshuffle of NaN payload must fail without --allow-nan"
    );
    let stderr = String::from_utf8_lossy(&status.stderr);
    assert!(
        stderr.contains("NaN") || stderr.contains("nan"),
        "error output must mention NaN: {stderr}"
    );
}

// ── --allow-nan flag unlocks the encode ────────────────────────────────────

#[test]
fn reshuffle_with_allow_nan_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.tgm");
    let output = dir.path().join("out.tgm");
    make_nan_file(&input);

    let status = cli()
        .args(["--allow-nan", "reshuffle", "-o"])
        .arg(&output)
        .arg(&input)
        .status()
        .unwrap();
    assert!(
        status.success(),
        "reshuffle --allow-nan must succeed on NaN payload"
    );
    assert!(output.exists());
}

// ── env var is equivalent to --allow-nan ───────────────────────────────────

#[test]
fn tensogram_allow_nan_env_var_unlocks_encode() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.tgm");
    let output = dir.path().join("out.tgm");
    make_nan_file(&input);

    let status = cli()
        .env("TENSOGRAM_ALLOW_NAN", "1")
        .args(["reshuffle", "-o"])
        .arg(&output)
        .arg(&input)
        .status()
        .unwrap();
    assert!(
        status.success(),
        "TENSOGRAM_ALLOW_NAN=1 must unlock encoding"
    );
}

// ── Regression: finite file round-trips unchanged with flag off/on ─────────

#[test]
fn reshuffle_finite_file_works_with_or_without_flag() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.tgm");
    let out_off = dir.path().join("out_off.tgm");
    let out_on = dir.path().join("out_on.tgm");
    make_finite_file(&input);

    // No flag.
    let s1 = cli()
        .args(["reshuffle", "-o"])
        .arg(&out_off)
        .arg(&input)
        .status()
        .unwrap();
    assert!(s1.success());

    // --allow-nan on (no-op for finite data).
    let s2 = cli()
        .args(["--allow-nan", "reshuffle", "-o"])
        .arg(&out_on)
        .arg(&input)
        .status()
        .unwrap();
    assert!(s2.success());
}

// ── Invalid mask-method name surfaces an error ─────────────────────────────

#[test]
fn unknown_nan_mask_method_fails_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.tgm");
    let output = dir.path().join("out.tgm");
    make_nan_file(&input);

    let status = cli()
        .args(["--allow-nan", "--nan-mask-method=bogus", "reshuffle", "-o"])
        .arg(&output)
        .arg(&input)
        .output()
        .unwrap();
    assert!(!status.status.success());
    let stderr = String::from_utf8_lossy(&status.stderr);
    assert!(
        stderr.contains("unknown mask method") || stderr.contains("bogus"),
        "error should mention unknown method, got: {stderr}"
    );
}
