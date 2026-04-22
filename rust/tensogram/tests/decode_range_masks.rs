// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Integration tests for mask-aware `decode_range`.
//! See `plans/WIRE_FORMAT.md` §6.5 for the wire-format spec.
//!
//! Verifies that partial-range decodes on payloads with NaN / Inf
//! masks restore canonical bit patterns only at positions that fall
//! inside the requested range.

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

fn decode_f64_vec(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

// ── Basic range: NaN inside the requested range ─────────────────────────────

#[test]
fn decode_range_restores_nan_when_position_falls_inside_range() {
    // Values: [1.0, NaN, 3.0, NaN, 5.0, 6.0, NaN, 8.0]
    let data = f64_bytes(&[1.0, f64::NAN, 3.0, f64::NAN, 5.0, 6.0, f64::NAN, 8.0]);
    let desc = make_descriptor(vec![8], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();

    // Range (2, 4) — elements 2..6: [3.0, NaN, 5.0, 6.0]
    let (_desc_out, parts) = decode_range(&msg, 0, &[(2, 4)], &DecodeOptions::default()).unwrap();
    let got = decode_f64_vec(&parts[0]);
    assert_eq!(got.len(), 4);
    assert_eq!(got[0], 3.0);
    assert!(got[1].is_nan());
    assert_eq!(got[2], 5.0);
    assert_eq!(got[3], 6.0);
}

// ── Range that skips NaNs entirely ──────────────────────────────────────────

#[test]
fn decode_range_without_any_masked_positions_returns_finite_values() {
    let data = f64_bytes(&[1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0, f64::NAN, 8.0]);
    let desc = make_descriptor(vec![8], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();

    // Range (2, 4) — elements 2..6: all finite [3.0, 4.0, 5.0, 6.0]
    let (_, parts) = decode_range(&msg, 0, &[(2, 4)], &DecodeOptions::default()).unwrap();
    let got = decode_f64_vec(&parts[0]);
    for (i, v) in got.iter().enumerate() {
        assert!(!v.is_nan(), "pos {i} must be finite, got {v}");
    }
    assert_eq!(got, vec![3.0, 4.0, 5.0, 6.0]);
}

// ── Multi-range request ─────────────────────────────────────────────────────

#[test]
fn decode_range_multiple_ranges_each_gets_correct_restoration() {
    let data = f64_bytes(&[
        f64::NAN,
        1.0,
        2.0,
        f64::INFINITY,
        4.0,
        5.0,
        f64::NEG_INFINITY,
        7.0,
    ]);
    let desc = make_descriptor(vec![8], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        allow_inf: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();

    // Range 1: (0, 2) → [NaN, 1.0]
    // Range 2: (2, 3) → [2.0, +Inf, 4.0]
    // Range 3: (5, 3) → [5.0, -Inf, 7.0]
    let (_, parts) = decode_range(
        &msg,
        0,
        &[(0, 2), (2, 3), (5, 3)],
        &DecodeOptions::default(),
    )
    .unwrap();
    assert_eq!(parts.len(), 3);

    let got0 = decode_f64_vec(&parts[0]);
    assert!(got0[0].is_nan());
    assert_eq!(got0[1], 1.0);

    let got1 = decode_f64_vec(&parts[1]);
    assert_eq!(got1[0], 2.0);
    assert!(got1[1].is_infinite() && got1[1].is_sign_positive());
    assert_eq!(got1[2], 4.0);

    let got2 = decode_f64_vec(&parts[2]);
    assert_eq!(got2[0], 5.0);
    assert!(got2[1].is_infinite() && got2[1].is_sign_negative());
    assert_eq!(got2[2], 7.0);
}

// ── restore_non_finite=false leaves zeros even for ranges ───────────────────

#[test]
fn decode_range_restore_off_returns_substituted_zeros() {
    let data = f64_bytes(&[1.0, f64::NAN, 3.0, 4.0]);
    let desc = make_descriptor(vec![4], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();

    let decode_opts = DecodeOptions {
        restore_non_finite: false,
        ..Default::default()
    };
    let (_, parts) = decode_range(&msg, 0, &[(0, 4)], &decode_opts).unwrap();
    let got = decode_f64_vec(&parts[0]);
    assert_eq!(got[1], 0.0, "element 1 must stay zero with restore off");
}

// ── Large sparse payload with ranges covering different regions ─────────────

#[test]
fn decode_range_large_sparse_payload() {
    let n = 5_000;
    let mut values: Vec<f64> = (0..n).map(|i| i as f64).collect();
    // NaN at every 101st position starting from 50.
    for i in (50..n).step_by(101) {
        values[i] = f64::NAN;
    }
    let data = f64_bytes(&values);
    let desc = make_descriptor(vec![n as u64], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();

    // Request 3 disjoint ranges covering different mask densities.
    let ranges = [(100u64, 500u64), (2000, 300), (4000, 1000)];
    let (_, parts) = decode_range(&msg, 0, &ranges, &DecodeOptions::default()).unwrap();

    // Verify each part element-by-element against the expected source values.
    for (i, &(offset, count)) in ranges.iter().enumerate() {
        let got = decode_f64_vec(&parts[i]);
        assert_eq!(got.len(), count as usize);
        for (j, &v) in got.iter().enumerate() {
            let global = offset as usize + j;
            let expected_nan = (global >= 50) && (global - 50).is_multiple_of(101);
            if expected_nan {
                assert!(
                    v.is_nan(),
                    "range {i} pos {j} (global {global}) should be NaN"
                );
            } else {
                assert_eq!(v, global as f64, "finite mismatch at range {i} pos {j}");
            }
        }
    }
}

// ── Non-masked payload: decode_range behaviour unchanged ────────────────────

#[test]
fn decode_range_without_masks_returns_expected_values() {
    // Sanity: existing decode_range behaviour is not altered by the
    // Commit 7 changes when the frame has no masks.
    let data = f64_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let desc = make_descriptor(vec![5], Dtype::Float64);
    let msg = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        },
    )
    .unwrap();

    let (_, parts) = decode_range(&msg, 0, &[(1, 3)], &DecodeOptions::default()).unwrap();
    let got = decode_f64_vec(&parts[0]);
    assert_eq!(got, vec![2.0, 3.0, 4.0]);
}
