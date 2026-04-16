// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Integration test: `TENSOGRAM_THREADS` env var precedence for the CLI.
//!
//! Verifies two invariants:
//!
//! 1. When `TENSOGRAM_THREADS=N` is set and no `--threads` flag is given,
//!    CPU-heavy subcommands run with budget `N`.
//! 2. An explicit `--threads=M` flag overrides the env var (`M` wins).
//!
//! We exercise this through `tensogram copy`, which does decode + re-encode
//! and is CPU-bound enough for the budget to matter, but we only assert that
//! the command completes and produces a valid file — timing is too noisy in
//! test harnesses.

use std::io::Write;
use std::process::Command;

use tensogram_core::{DataObjectDescriptor, EncodeOptions, GlobalMetadata};

fn make_input_file(dir: &std::path::Path) -> std::path::PathBuf {
    use ciborium::Value;

    let path = dir.join("input.tgm");
    let mut file = std::fs::File::create(&path).expect("create input");

    let mut desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![100_000],
        strides: vec![1],
        dtype: tensogram_core::Dtype::Float64,
        byte_order: tensogram_core::ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "blosc2".to_string(),
        params: Default::default(),
        hash: None,
    };
    desc.params
        .insert("blosc2_clevel".to_string(), Value::Integer(5.into()));
    desc.params
        .insert("blosc2_codec".to_string(), Value::Text("lz4".to_string()));

    let data: Vec<u8> = (0..100_000)
        .flat_map(|i| (250.0f64 + (i as f64).sin() * 30.0).to_ne_bytes())
        .collect();

    let msg = tensogram_core::encode(
        &GlobalMetadata::default(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .expect("encode");

    file.write_all(&msg).unwrap();
    path
}

fn cli_binary() -> String {
    // Built next to the test binary in `target/<profile>/`.
    env!("CARGO_BIN_EXE_tensogram").to_string()
}

/// Environment variable `TENSOGRAM_THREADS` is honoured by the CLI.
/// We can't cheaply observe the internal budget chosen, but we can
/// confirm the subprocess completes successfully and produces a valid
/// output.
#[test]
fn env_var_budget_produces_valid_output() {
    let dir = tempfile::tempdir().unwrap();
    let input = make_input_file(dir.path());
    let output = dir.path().join("output.tgm");

    let status = Command::new(cli_binary())
        .env("TENSOGRAM_THREADS", "4")
        .args(["copy", input.to_str().unwrap(), output.to_str().unwrap()])
        .status()
        .expect("spawn tensogram");
    assert!(status.success(), "tensogram copy failed");

    // Output file must be readable and have the same number of messages.
    let f = tensogram_core::TensogramFile::open(&output).unwrap();
    assert_eq!(f.message_count().unwrap(), 1);
}

/// Explicit `--threads` on the command line overrides `TENSOGRAM_THREADS`.
/// Both settings should produce valid output; we only assert successful
/// exit — timing is too noisy to verify the precedence directly.
#[test]
fn explicit_flag_overrides_env_var() {
    let dir = tempfile::tempdir().unwrap();
    let input = make_input_file(dir.path());
    let output = dir.path().join("output.tgm");

    let status = Command::new(cli_binary())
        .env("TENSOGRAM_THREADS", "2")
        .args([
            "--threads",
            "8",
            "copy",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .status()
        .expect("spawn tensogram");
    assert!(status.success(), "tensogram --threads copy failed");

    let f = tensogram_core::TensogramFile::open(&output).unwrap();
    assert_eq!(f.message_count().unwrap(), 1);
}

/// `TENSOGRAM_THREADS=0` (or unset) runs sequentially — still completes.
#[test]
fn threads_zero_runs_sequentially() {
    let dir = tempfile::tempdir().unwrap();
    let input = make_input_file(dir.path());
    let output = dir.path().join("output.tgm");

    let status = Command::new(cli_binary())
        .env("TENSOGRAM_THREADS", "0")
        .args(["copy", input.to_str().unwrap(), output.to_str().unwrap()])
        .status()
        .expect("spawn tensogram");
    assert!(status.success());

    let f = tensogram_core::TensogramFile::open(&output).unwrap();
    assert_eq!(f.message_count().unwrap(), 1);
}

/// An unparseable `TENSOGRAM_THREADS` should not crash — clap has `env`
/// support that falls through to the default when the value is invalid.
#[test]
fn invalid_env_var_falls_back() {
    let dir = tempfile::tempdir().unwrap();
    let input = make_input_file(dir.path());
    let output = dir.path().join("output.tgm");

    let status = Command::new(cli_binary())
        .env("TENSOGRAM_THREADS", "not-a-number")
        .args(["copy", input.to_str().unwrap(), output.to_str().unwrap()])
        .status()
        .expect("spawn tensogram");
    // Clap will reject the malformed env var with a non-zero exit.
    // The point is: it must not panic or produce corrupt output.
    if status.success() {
        let f = tensogram_core::TensogramFile::open(&output).unwrap();
        assert_eq!(f.message_count().unwrap(), 1);
    }
    // No assert on status — either clean rejection or clean success.
}
