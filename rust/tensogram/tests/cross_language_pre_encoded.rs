// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Cross-language SHA256 golden test for `encode_pre_encoded()`.
//!
//! This test verifies that encoding the **same** pre-encoded payload via Rust,
//! Python, and C++ `encode_pre_encoded()` produces decoded payloads whose
//! SHA-256 hashes are identical. Because provenance fields (UUID, timestamp)
//! differ across encode calls, we compare decoded-payload hashes, **not** raw
//! wire bytes.
//!
//! The test driver:
//! 1. Generates a deterministic float64[1024] input.
//! 2. Encodes via Rust `encode_pre_encoded()` → decodes → SHA-256.
//! 3. Spawns a Python helper that does the same → reads its SHA-256 from stdout.
//! 4. Spawns a C++ helper that does the same → reads its SHA-256 from stdout.
//! 5. Asserts all three SHA-256 values are equal.
//!
//! The test is gated: if the Python venv or C++ helper binary is unavailable
//! the corresponding sub-test is skipped with a warning (not failed).

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;

use tensogram::{
    ByteOrder, DataObjectDescriptor, Dtype, EncodeOptions, GlobalMetadata, decode,
    encode_pre_encoded,
};

// ── Deterministic input ────────────────────────────────────────────────────

/// Generate 1024 deterministic float64 values.
fn deterministic_f64_1024() -> Vec<f64> {
    (0..1024).map(|i| 200.0 + (i as f64) * 0.125).collect()
}

/// Little-endian bytes for a slice of f64.
fn f64_to_le_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// SHA-256 hex string of a byte slice.
fn sha256_hex(data: &[u8]) -> String {
    let hash = Sha256::digest(data);
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

// ── Rust encode + decode ───────────────────────────────────────────────────

/// Encode the deterministic payload via Rust `encode_pre_encoded`, decode,
/// and return the SHA-256 hex string of the decoded payload bytes.
fn rust_encode_decode_sha256() -> String {
    let values = deterministic_f64_1024();
    let raw_bytes = f64_to_le_bytes(&values);

    // Build descriptor for encoding="none" (raw pass-through).
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![1024],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    };

    let meta = GlobalMetadata::default();
    let opts = EncodeOptions::default();

    let wire =
        encode_pre_encoded(&meta, &[(&desc, &raw_bytes)], &opts).expect("Rust encode_pre_encoded");

    // Decode and extract the payload.
    let (_meta, objects) = decode(&wire, &Default::default()).expect("Rust decode");
    assert_eq!(objects.len(), 1);
    let (_desc, payload) = &objects[0];
    sha256_hex(payload)
}

// ── Helper spawning ────────────────────────────────────────────────────────

/// Find the project root from CARGO_MANIFEST_DIR.
fn project_root() -> PathBuf {
    // CARGO_MANIFEST_DIR for this crate is crates/tensogram
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("project root")
        .to_path_buf()
}

/// Spawn the Python helper; returns Ok(sha256_hex) or Err(skip reason).
fn python_sha256() -> Result<String, String> {
    let root = project_root();
    let venv_python = root.join(".venv/bin/python");
    if !venv_python.exists() {
        return Err(format!(
            "Python venv not found at {}",
            venv_python.display()
        ));
    }

    let helper = root.join("tests/python/cross_language_pre_encoded_helper.py");
    if !helper.exists() {
        return Err(format!("Python helper not found at {}", helper.display()));
    }

    let output = std::process::Command::new(&venv_python)
        .arg(&helper)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .map_err(|e| format!("Failed to spawn Python: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Python helper failed: {stderr}"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let sha = stdout.trim().to_string();
    if sha.len() != 64 {
        return Err(format!("Python output not a valid SHA-256 hex: '{sha}'"));
    }
    Ok(sha)
}

/// Spawn the C++ helper; returns Ok(sha256_hex) or Err(skip reason).
fn cpp_sha256() -> Result<String, String> {
    let root = project_root();
    let helper = root.join("build/cross_language_pre_encoded_helper");
    if !helper.exists() {
        return Err(format!(
            "C++ helper not found at {}; build it first",
            helper.display()
        ));
    }

    let library_dir = root.join("target/release");
    let append_library_dir = |var_name: &str| -> Result<std::ffi::OsString, String> {
        let mut paths = vec![library_dir.clone()];
        if let Some(existing) = std::env::var_os(var_name) {
            paths.extend(std::env::split_paths(&existing));
        }
        std::env::join_paths(paths)
            .map_err(|e| format!("Failed to construct {var_name} for C++ helper: {e}"))
    };

    let mut command = std::process::Command::new(&helper);
    command.env(
        "DYLD_LIBRARY_PATH",
        append_library_dir("DYLD_LIBRARY_PATH")?,
    );
    command.env("LD_LIBRARY_PATH", append_library_dir("LD_LIBRARY_PATH")?);

    let output = command
        .output()
        .map_err(|e| format!("Failed to spawn C++ helper: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("C++ helper failed: {stderr}"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let sha = stdout.trim().to_string();
    if sha.len() != 64 {
        return Err(format!("C++ output not a valid SHA-256 hex: '{sha}'"));
    }
    Ok(sha)
}

// ── Test ────────────────────────────────────────────────────────────────────

#[test]
fn cross_language_pre_encoded_sha256() {
    let rust_sha = rust_encode_decode_sha256();
    eprintln!("Rust  SHA-256: {rust_sha}");

    // Python
    match python_sha256() {
        Ok(py_sha) => {
            eprintln!("Python SHA-256: {py_sha}");
            assert_eq!(
                rust_sha, py_sha,
                "Rust and Python decoded-payload SHA-256 differ"
            );
        }
        Err(reason) => {
            eprintln!("SKIP Python: {reason}");
        }
    }

    // C++
    match cpp_sha256() {
        Ok(cpp_sha) => {
            eprintln!("C++   SHA-256: {cpp_sha}");
            assert_eq!(
                rust_sha, cpp_sha,
                "Rust and C++ decoded-payload SHA-256 differ"
            );
        }
        Err(reason) => {
            eprintln!("SKIP C++: {reason}");
        }
    }
}
