// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Integration tests for mask-aware `validate --full` (Commit 8 of
//! `plans/BITMASK_FRAME.md`).
//!
//! Covers:
//! - Masked NaN/Inf positions do NOT raise NanDetected / InfDetected
//!   (they are expected).
//! - Unexpected NaN / Inf at non-masked positions still errors.
//! - Validation passes when no masks are present (legacy behaviour
//!   preserved).

use std::collections::BTreeMap;
use tensogram::*;

fn make_global_meta() -> GlobalMetadata {
    GlobalMetadata {
        version: 3,
        ..Default::default()
    }
}

fn make_descriptor(shape: Vec<u64>, dtype: Dtype) -> DataObjectDescriptor {
    let strides = if shape.is_empty() {
        vec![]
    } else {
        let mut s = vec![1u64; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            s[i] = s[i + 1] * shape[i + 1];
        }
        s
    };
    DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: shape.len() as u64,
        shape,
        strides,
        dtype,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        masks: None,
        params: BTreeMap::new(),
    }
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

// ── Mask-bearing frames pass fidelity validation ────────────────────────────

#[test]
fn validate_full_passes_on_masked_nan_frame() {
    // A frame with NaN recorded via allow_nan mask validates cleanly
    // at Fidelity level — the NaN at the masked position is expected.
    let data = f64_bytes(&[1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
    let desc = make_descriptor(vec![5], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();

    let report = validate_message(
        &msg,
        &ValidateOptions {
            max_level: ValidationLevel::Fidelity,
            ..Default::default()
        },
    );
    // Count any NaN/Inf errors — masked positions must NOT show up.
    let nan_errors: Vec<_> = report
        .issues
        .iter()
        .filter(|i| matches!(i.code, IssueCode::NanDetected | IssueCode::InfDetected))
        .collect();
    assert!(
        nan_errors.is_empty(),
        "mask-bearing frame must not raise NaN/Inf validation errors at Fidelity level; got: {nan_errors:?}"
    );
}

#[test]
fn validate_full_passes_on_masked_inf_frame() {
    let data = f64_bytes(&[f64::INFINITY, 1.0, f64::NEG_INFINITY, 2.0]);
    let desc = make_descriptor(vec![4], Dtype::Float64);
    let options = EncodeOptions {
        allow_inf: true,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();

    let report = validate_message(
        &msg,
        &ValidateOptions {
            max_level: ValidationLevel::Fidelity,
            ..Default::default()
        },
    );
    let nan_errors: Vec<_> = report
        .issues
        .iter()
        .filter(|i| matches!(i.code, IssueCode::NanDetected | IssueCode::InfDetected))
        .collect();
    assert!(
        nan_errors.is_empty(),
        "+Inf/-Inf at masked positions must not raise validation errors; got: {nan_errors:?}"
    );
}

// ── Legacy: non-masked NaN still raises error ──────────────────────────────

#[test]
fn validate_full_rejects_non_finite_without_masks() {
    // Build a frame by hand with NaN in the payload but NO masks
    // sub-map — this simulates either a corrupted file or a
    // pre-0.17-produced NTensorFrame that somehow passes NaN through.
    // The validator must still flag it as NanDetected.
    //
    // We use encode_pre_encoded with a NaN payload and no allow_nan,
    // which produces a type-9 frame with no masks.
    let data = f64_bytes(&[1.0, f64::NAN, 3.0]);
    let desc = make_descriptor(vec![3], Dtype::Float64);
    let msg = encode_pre_encoded(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .unwrap();

    let report = validate_message(
        &msg,
        &ValidateOptions {
            max_level: ValidationLevel::Fidelity,
            ..Default::default()
        },
    );
    // The NaN at position 1 is unexpected (no mask) — must raise.
    let has_nan_error = report
        .issues
        .iter()
        .any(|i| matches!(i.code, IssueCode::NanDetected));
    assert!(
        has_nan_error,
        "non-masked NaN must still raise NanDetected at Fidelity level; issues: {:?}",
        report.issues
    );
}

// ── Finite-only mask-bearing frame still validates ──────────────────────────

#[test]
fn validate_full_passes_on_masked_frame_with_all_finite_payload() {
    // Edge: allow_nan=true + finite input → no masks sub-map emitted,
    // frame validates like any other finite frame.
    let data = f64_bytes(&[1.0, 2.0, 3.0]);
    let desc = make_descriptor(vec![3], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        allow_inf: true,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();

    let report = validate_message(
        &msg,
        &ValidateOptions {
            max_level: ValidationLevel::Fidelity,
            ..Default::default()
        },
    );
    let errors: Vec<_> = report
        .issues
        .iter()
        .filter(|i| i.severity == IssueSeverity::Error)
        .collect();
    assert!(
        errors.is_empty(),
        "finite-only frame must pass; errors: {errors:?}"
    );
}

// ── Multi-object: only non-masked NaN fires ────────────────────────────────

#[test]
fn validate_full_multi_object_only_flags_unexpected_positions() {
    // Object 0: masked NaN — should not fire.
    // Object 1: finite — should not fire.
    let desc = make_descriptor(vec![3], Dtype::Float64);
    let d0 = f64_bytes(&[f64::NAN, 1.0, 2.0]);
    let d1 = f64_bytes(&[3.0, 4.0, 5.0]);

    let options = EncodeOptions {
        allow_nan: true,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &d0), (&desc, &d1)], &options).unwrap();

    let report = validate_message(
        &msg,
        &ValidateOptions {
            max_level: ValidationLevel::Fidelity,
            ..Default::default()
        },
    );
    let nan_errors: Vec<_> = report
        .issues
        .iter()
        .filter(|i| matches!(i.code, IssueCode::NanDetected | IssueCode::InfDetected))
        .collect();
    assert!(
        nan_errors.is_empty(),
        "all NaNs are masked; no errors expected; got: {nan_errors:?}"
    );
}
