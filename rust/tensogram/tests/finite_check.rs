// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Integration tests pinning the 0.17 default-reject-non-finite
//! behaviour at the `tensogram::encode` entry point.
//!
//! Unlike the pre-0.17 `EncodeOptions::reject_nan` / `reject_inf`
//! flags (removed in commit 0 of the bitmask frame work), rejection
//! is now always on by default.  The `allow_nan` / `allow_inf`
//! bitmask opt-in (Commit 5) flips the policy for callers who want
//! their non-finite values preserved via a mask companion frame;
//! tests for that path live in `tests/allow_nan_inf.rs`.
//!
//! Cross-references:
//! - Design: `docs/src/guide/nan-inf-handling.md`.
//! - Wire-format spec: `plans/WIRE_FORMAT.md` §6.5.
//! - Substitution internals: `rust/tensogram/src/substitute_and_mask.rs`.

use std::collections::BTreeMap;
use tensogram::*;

fn make_global_meta() -> GlobalMetadata {
    GlobalMetadata {
        version: 3,
        ..Default::default()
    }
}

fn make_descriptor(shape: Vec<u64>, dtype: Dtype, byte_order: ByteOrder) -> DataObjectDescriptor {
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
        byte_order,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

// ── Default-reject behaviour for every float dtype ──────────────────────────

#[test]
fn default_encode_rejects_nan_float32() {
    let data = f32_bytes(&[1.0, f32::NAN, 3.0]);
    let desc = make_descriptor(vec![3], Dtype::Float32, ByteOrder::native());
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(matches!(err, TensogramError::Encoding(_)));
    assert!(msg.contains("NaN"), "message must name the kind: {msg}");
    assert!(
        msg.contains("element 1"),
        "message must name the index: {msg}"
    );
    assert!(
        msg.contains("float32"),
        "message must name the dtype: {msg}"
    );
}

#[test]
fn default_encode_rejects_positive_inf_float64() {
    let data = f64_bytes(&[1.0, f64::INFINITY]);
    let desc = make_descriptor(vec![2], Dtype::Float64, ByteOrder::native());
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("+Inf"));
}

#[test]
fn default_encode_rejects_negative_inf_float64() {
    let data = f64_bytes(&[1.0, f64::NEG_INFINITY]);
    let desc = make_descriptor(vec![2], Dtype::Float64, ByteOrder::native());
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("-Inf"));
}

#[test]
fn default_encode_rejects_complex64_nan_in_real() {
    let data: Vec<u8> = [1.0_f32, 2.0, f32::NAN, 3.0]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    let desc = make_descriptor(vec![2], Dtype::Complex64, ByteOrder::native());
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("NaN") && msg.contains("real"));
}

#[test]
fn default_encode_rejects_float16_nan_bit_level() {
    // IEEE half: exp == 0x1F, mantissa != 0 → NaN.  Pattern 0x7E00.
    let data: Vec<u8> = [0x3C00u16, 0x7E00]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    let desc = make_descriptor(vec![2], Dtype::Float16, ByteOrder::native());
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("NaN"));
}

// ── Finite input still encodes successfully ─────────────────────────────────

#[test]
fn default_encode_accepts_all_finite_float64() {
    let data = f64_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let desc = make_descriptor(vec![4], Dtype::Float64, ByteOrder::native());
    encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .expect("finite data must encode");
}

#[test]
fn default_encode_accepts_negative_zero_and_subnormals() {
    // Edge-case values that are finite: -0.0 and a subnormal f64.
    let subnormal = f64::from_bits(0x0000_0000_0000_0001);
    let data = f64_bytes(&[0.0, -0.0, subnormal, -subnormal]);
    let desc = make_descriptor(vec![4], Dtype::Float64, ByteOrder::native());
    encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .expect("-0.0 and subnormals are finite, must pass");
}

// ── Integer and bitmask dtypes are never scanned ────────────────────────────

#[test]
fn default_encode_never_scans_integer_dtypes() {
    // Even bit patterns that WOULD be NaN as f32 are accepted when the
    // dtype says uint32.
    let data = vec![0xFFu8; 16]; // 4 u32 elements, all 0xFFFFFFFF
    let desc = make_descriptor(vec![4], Dtype::Uint32, ByteOrder::native());
    encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .expect("uint32 payload must encode regardless of bit pattern");
}

// ── encode_pre_encoded is NOT gated by finite-check ─────────────────────────

#[test]
fn encode_pre_encoded_accepts_opaque_nan_bytes() {
    // Pre-encoded bytes are opaque; caller owns the contract.
    let data = f32_bytes(&[1.0, f32::NAN]);
    let desc = make_descriptor(vec![2], Dtype::Float32, ByteOrder::native());
    encode_pre_encoded(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .expect("encode_pre_encoded must not run the finite check");
}
