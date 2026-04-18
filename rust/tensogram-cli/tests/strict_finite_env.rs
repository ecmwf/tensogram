// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Integration tests for the CLI `--reject-nan` / `--reject-inf` flags
//! and their `TENSOGRAM_REJECT_{NAN,INF}` env-var fallbacks.
//!
//! We exercise them through `tensogram merge`, which is the cheapest
//! encoding-capable subcommand that re-runs the pipeline on decoded data.

use std::io::Write;
use std::process::Command;

use tensogram::{DataObjectDescriptor, EncodeOptions, GlobalMetadata};

fn cli_binary() -> String {
    env!("CARGO_BIN_EXE_tensogram").to_string()
}

/// Build a .tgm file containing one float64 message whose payload has
/// a NaN at element 5.  Used to exercise the strict flags end-to-end.
fn make_nan_file(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("nan_input.tgm");
    let mut file = std::fs::File::create(&path).unwrap();

    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![10],
        strides: vec![1],
        dtype: tensogram::Dtype::Float64,
        byte_order: tensogram::ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: Default::default(),
        hash: None,
    };

    let mut values = [1.0_f64; 10];
    values[5] = f64::NAN;
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let msg = tensogram::encode(
        &GlobalMetadata::default(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .expect("encode NaN file");
    file.write_all(&msg).unwrap();
    path
}

/// Same but with +Inf at element 5.
fn make_inf_file(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("inf_input.tgm");
    let mut file = std::fs::File::create(&path).unwrap();

    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![10],
        strides: vec![1],
        dtype: tensogram::Dtype::Float64,
        byte_order: tensogram::ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: Default::default(),
        hash: None,
    };

    let mut values = [1.0_f64; 10];
    values[5] = f64::INFINITY;
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let msg = tensogram::encode(
        &GlobalMetadata::default(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .expect("encode Inf file");
    file.write_all(&msg).unwrap();
    path
}

// ── Default behaviour: merge succeeds with NaN-bearing input ──────────────

#[test]
fn merge_without_flags_accepts_nan() {
    let dir = tempfile::tempdir().unwrap();
    let input = make_nan_file(dir.path());
    let output = dir.path().join("out.tgm");
    let status = Command::new(cli_binary())
        .args([
            "merge",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .status()
        .expect("spawn tensogram");
    assert!(status.success(), "default merge must accept NaN");
}

// ── --reject-nan flag fails on NaN input ──────────────────────────────────

#[test]
fn merge_with_reject_nan_flag_fails_on_nan() {
    let dir = tempfile::tempdir().unwrap();
    let input = make_nan_file(dir.path());
    let output = dir.path().join("out.tgm");
    let out = Command::new(cli_binary())
        .args([
            "--reject-nan",
            "merge",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .output()
        .expect("spawn tensogram");
    assert!(
        !out.status.success(),
        "merge with --reject-nan must fail on NaN input"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.to_lowercase().contains("nan"),
        "expected NaN error in stderr, got: {stderr}"
    );
}

// ── --reject-inf flag fails on Inf input ──────────────────────────────────

#[test]
fn merge_with_reject_inf_flag_fails_on_inf() {
    let dir = tempfile::tempdir().unwrap();
    let input = make_inf_file(dir.path());
    let output = dir.path().join("out.tgm");
    let out = Command::new(cli_binary())
        .args([
            "--reject-inf",
            "merge",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .output()
        .expect("spawn tensogram");
    assert!(
        !out.status.success(),
        "merge with --reject-inf must fail on Inf input"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.to_lowercase().contains("inf"),
        "expected Inf error in stderr, got: {stderr}"
    );
}

// ── Env var fallback ──────────────────────────────────────────────────────

#[test]
fn tensogram_reject_nan_env_var_honoured() {
    let dir = tempfile::tempdir().unwrap();
    let input = make_nan_file(dir.path());
    let output = dir.path().join("out.tgm");
    let out = Command::new(cli_binary())
        .env("TENSOGRAM_REJECT_NAN", "1")
        .args([
            "merge",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .output()
        .expect("spawn tensogram");
    assert!(
        !out.status.success(),
        "merge with TENSOGRAM_REJECT_NAN=1 must fail on NaN input"
    );
}

#[test]
fn tensogram_reject_inf_env_var_honoured() {
    let dir = tempfile::tempdir().unwrap();
    let input = make_inf_file(dir.path());
    let output = dir.path().join("out.tgm");
    let out = Command::new(cli_binary())
        .env("TENSOGRAM_REJECT_INF", "1")
        .args([
            "merge",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .output()
        .expect("spawn tensogram");
    assert!(
        !out.status.success(),
        "merge with TENSOGRAM_REJECT_INF=1 must fail on Inf input"
    );
}

// ── Env var = 0 / false is explicit opt-out ─────────────────────────────

#[test]
fn tensogram_reject_nan_env_var_zero_disables() {
    // `TENSOGRAM_REJECT_NAN=0` must be treated as off, not on.
    // Clap's bool env parsing: "0" / "false" → false, "1" / "true" → true.
    let dir = tempfile::tempdir().unwrap();
    let input = make_nan_file(dir.path());
    let output = dir.path().join("out.tgm");
    let status = Command::new(cli_binary())
        .env("TENSOGRAM_REJECT_NAN", "0")
        .args([
            "merge",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .status()
        .expect("spawn tensogram");
    assert!(
        status.success(),
        "merge with TENSOGRAM_REJECT_NAN=0 must accept NaN (explicit opt-out)"
    );
}

// ── Orthogonality — reject_inf does not fail on NaN ───────────────────────

#[test]
fn reject_inf_does_not_fail_on_nan_input() {
    let dir = tempfile::tempdir().unwrap();
    let input = make_nan_file(dir.path());
    let output = dir.path().join("out.tgm");
    let status = Command::new(cli_binary())
        .args([
            "--reject-inf",
            "merge",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .status()
        .expect("spawn tensogram");
    assert!(status.success(), "--reject-inf alone must not catch NaN");
}
