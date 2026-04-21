// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Example 13 — Message validation
//!
//! Demonstrates the validation API for checking Tensogram messages
//! at different levels of depth.

use std::collections::BTreeMap;

use tensogram::{
    ByteOrder, DataObjectDescriptor, Dtype, EncodeOptions, GlobalMetadata, ValidateOptions,
    ValidationLevel, encode, validate_message,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Encode a valid message ─────────────────────────────────────────────

    let meta = GlobalMetadata::default();
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
    let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();

    let msg = encode(
        &meta,
        &[(&desc, data.as_slice())],
        &EncodeOptions::default(),
    )?;

    // ── 2. Validate at default level (integrity) ─────────────────────────────

    println!("=== Default validation ===\n");
    let report = validate_message(&msg, &ValidateOptions::default());
    println!(
        "  issues: {}, objects: {}, hash_verified: {}",
        report.issues.len(),
        report.object_count,
        report.hash_verified
    );
    assert!(report.is_ok());

    // ── 3. Quick mode (structure only) ───────────────────────────────────────

    println!("\n=== Quick validation ===\n");
    let quick_opts = ValidateOptions {
        max_level: ValidationLevel::Structure,
        ..ValidateOptions::default()
    };
    let report = validate_message(&msg, &quick_opts);
    println!(
        "  issues: {}, hash_verified: {} (skipped in quick mode)",
        report.issues.len(),
        report.hash_verified
    );

    // ── 4. Full mode (fidelity — NaN/Inf detection) ──────────────────────────

    println!("\n=== Full validation ===\n");
    let full_opts = ValidateOptions {
        max_level: ValidationLevel::Fidelity,
        ..ValidateOptions::default()
    };
    let report = validate_message(&msg, &full_opts);
    println!(
        "  issues: {}, hash_verified: {}",
        report.issues.len(),
        report.hash_verified
    );

    // ── 5. Detect corrupted message ──────────────────────────────────────────

    println!("\n=== Corrupted message ===\n");
    let mut corrupted = msg.clone();
    corrupted[0..8].copy_from_slice(b"WRONGMAG");
    let report = validate_message(&corrupted, &ValidateOptions::default());
    for issue in &report.issues {
        println!(
            "  [{:?}] {:?}: {}",
            issue.severity, issue.code, issue.description
        );
    }

    // ── 6. JSON output ───────────────────────────────────────────────────────

    println!("\n=== JSON report ===\n");
    let report = validate_message(&msg, &ValidateOptions::default());
    let json = serde_json::to_string_pretty(&report)?;
    println!("{json}");

    Ok(())
}
