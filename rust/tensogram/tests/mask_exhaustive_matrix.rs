// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Exhaustive test matrix for the NaN / Inf bitmask companion frame
//! (Commit 14 of `plans/BITMASK_FRAME.md` §11).
//!
//! Matrix axes:
//! - dtype: f64, c64 (sampled: f16, bf16, f32, c128 in separate tests)
//! - input shape: no-non-finite / nan-only / pos_inf-only / neg_inf-only /
//!   all-three-kinds
//! - mask method: none / rle / roaring / lz4 / zstd / blosc2 (exhaustive)
//! - decode path: decode / decode_with_masks / decode_range(single) /
//!   decode_range(multi)
//!
//! Other axes (encoding, filter, compression) use one sensible combination
//! per dimension, covered by smaller focused test-cases.

use std::collections::BTreeMap;
use tensogram::encode::MaskMethod;
use tensogram::*;

// ── Helpers ────────────────────────────────────────────────────────────────

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
        hash: None,
    }
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// Enumerate the five input-shape variants for f64, each 8 elements long.
fn f64_input_shapes() -> Vec<(&'static str, Vec<f64>)> {
    vec![
        ("no-nonfinite", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        (
            "nan-only",
            vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0, 6.0, f64::NAN, 8.0],
        ),
        (
            "pos_inf-only",
            vec![1.0, f64::INFINITY, 3.0, f64::INFINITY, 5.0, 6.0, 7.0, 8.0],
        ),
        (
            "neg_inf-only",
            vec![
                1.0,
                f64::NEG_INFINITY,
                3.0,
                4.0,
                5.0,
                f64::NEG_INFINITY,
                7.0,
                8.0,
            ],
        ),
        (
            "all-three-kinds",
            vec![
                f64::NAN,
                1.0,
                f64::INFINITY,
                3.0,
                f64::NEG_INFINITY,
                5.0,
                f64::NAN,
                7.0,
            ],
        ),
    ]
}

/// Every mask method supported by the build (blosc2 gated on feature).
fn all_mask_methods() -> Vec<(&'static str, MaskMethod)> {
    let mut methods = vec![
        ("none", MaskMethod::None),
        ("rle", MaskMethod::Rle),
        ("roaring", MaskMethod::Roaring),
    ];
    #[cfg(feature = "lz4")]
    methods.push(("lz4", MaskMethod::Lz4));
    #[cfg(feature = "zstd")]
    methods.push(("zstd", MaskMethod::Zstd { level: None }));
    #[cfg(feature = "blosc2")]
    methods.push((
        "blosc2",
        MaskMethod::Blosc2 {
            codec: tensogram_encodings::pipeline::Blosc2Codec::Lz4,
            level: 5,
        },
    ));
    methods
}

/// Assert decoded f64 output matches the expected logical values.
/// Non-finite comparison is kind-based (NaN == NaN; +Inf == +Inf).
fn assert_f64_matches(got: &[u8], expected: &[f64], label: &str) {
    let got: Vec<f64> = got
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(
        got.len(),
        expected.len(),
        "[{label}] length mismatch: {} vs {}",
        got.len(),
        expected.len(),
    );
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() {
            assert!(g.is_nan(), "[{label}] pos {i}: expected NaN, got {g}");
        } else if e.is_infinite() {
            assert!(
                g.is_infinite() && g.signum() == e.signum(),
                "[{label}] pos {i}: expected {e}, got {g}",
            );
        } else {
            assert_eq!(g, e, "[{label}] pos {i}");
        }
    }
}

// ── f64 exhaustive test matrix ─────────────────────────────────────────────

#[test]
fn f64_exhaustive_encoding_none_filter_none_compression_none() {
    for (shape_name, values) in f64_input_shapes() {
        for (method_name, method) in all_mask_methods() {
            let label = format!("f64/none/{shape_name}/nan_method={method_name}");
            let desc = make_descriptor(vec![values.len() as u64], Dtype::Float64);
            let options = EncodeOptions {
                allow_nan: true,
                allow_inf: true,
                hash_algorithm: None,
                nan_mask_method: method.clone(),
                pos_inf_mask_method: method.clone(),
                neg_inf_mask_method: method.clone(),
                small_mask_threshold_bytes: 0,
                ..Default::default()
            };
            let data = f64_bytes(&values);
            let msg = encode(&make_global_meta(), &[(&desc, &data)], &options)
                .unwrap_or_else(|e| panic!("[{label}] encode failed: {e}"));

            // decode path
            let (_, objects) = decode(&msg, &DecodeOptions::default())
                .unwrap_or_else(|e| panic!("[{label}] decode failed: {e}"));
            assert_eq!(objects.len(), 1, "[{label}]");
            assert_f64_matches(&objects[0].1, &values, &format!("{label}/decode"));

            // decode_with_masks path — payload has zeros at masked positions.
            let (_, with_masks) = decode_with_masks(&msg, &DecodeOptions::default())
                .unwrap_or_else(|e| panic!("[{label}] decode_with_masks failed: {e}"));
            assert_eq!(with_masks.len(), 1);
            let substituted_expected: Vec<f64> = values
                .iter()
                .map(|v| if v.is_finite() { *v } else { 0.0 })
                .collect();
            assert_f64_matches(
                &with_masks[0].payload,
                &substituted_expected,
                &format!("{label}/decode_with_masks"),
            );

            // Masked bits must match the kind distribution in `values`.
            let nan_indices: Vec<bool> = values.iter().map(|v| v.is_nan()).collect();
            let pos_inf_indices: Vec<bool> = values
                .iter()
                .map(|v| v.is_infinite() && v.is_sign_positive())
                .collect();
            let neg_inf_indices: Vec<bool> = values
                .iter()
                .map(|v| v.is_infinite() && v.is_sign_negative())
                .collect();
            let any_nan = nan_indices.iter().any(|&b| b);
            let any_pos = pos_inf_indices.iter().any(|&b| b);
            let any_neg = neg_inf_indices.iter().any(|&b| b);
            assert_eq!(
                with_masks[0].masks.nan.is_some(),
                any_nan,
                "[{label}] nan mask presence"
            );
            assert_eq!(
                with_masks[0].masks.pos_inf.is_some(),
                any_pos,
                "[{label}] pos_inf mask presence"
            );
            assert_eq!(
                with_masks[0].masks.neg_inf.is_some(),
                any_neg,
                "[{label}] neg_inf mask presence"
            );

            // decode_range single range path — middle 4 elements.
            let (_, parts) = decode_range(&msg, 0, &[(2, 4)], &DecodeOptions::default())
                .unwrap_or_else(|e| panic!("[{label}] decode_range failed: {e}"));
            assert_eq!(parts.len(), 1);
            assert_f64_matches(&parts[0], &values[2..6], &format!("{label}/range"));

            // decode_range multi ranges path — two disjoint ranges.
            let (_, parts_multi) =
                decode_range(&msg, 0, &[(0, 3), (5, 3)], &DecodeOptions::default())
                    .unwrap_or_else(|e| panic!("[{label}] decode_range multi failed: {e}"));
            assert_eq!(parts_multi.len(), 2);
            assert_f64_matches(
                &parts_multi[0],
                &values[0..3],
                &format!("{label}/range_multi_0"),
            );
            assert_f64_matches(
                &parts_multi[1],
                &values[5..8],
                &format!("{label}/range_multi_1"),
            );
        }
    }
}

// ── f64 × simple_packing (substitute-and-mask runs before the codec) ──────

#[test]
fn f64_simple_packing_with_nan_round_trip() {
    // simple_packing cannot represent non-finite values; substitute_and_mask
    // replaces them with 0.0 BEFORE compute_params runs.  Decode restores
    // canonical NaN afterwards.
    let mut values: Vec<f64> = (0..32).map(|i| 10.0 + i as f64 * 0.5).collect();
    values[5] = f64::NAN;
    values[20] = f64::NAN;

    let mut desc = make_descriptor(vec![32], Dtype::Float64);
    desc.encoding = "simple_packing".to_string();
    desc.params
        .insert("reference_value".to_string(), ciborium::Value::Float(10.0));
    desc.params.insert(
        "binary_scale_factor".to_string(),
        ciborium::Value::Integer(0.into()),
    );
    desc.params.insert(
        "decimal_scale_factor".to_string(),
        ciborium::Value::Integer(0.into()),
    );
    desc.params.insert(
        "bits_per_value".to_string(),
        ciborium::Value::Integer(16.into()),
    );

    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let data = f64_bytes(&values);
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();

    let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    let got: Vec<f64> = objects[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert!(got[5].is_nan());
    assert!(got[20].is_nan());
    // Finite values round-trip through simple_packing with ≤ 0.5 LSB loss.
    // bits_per_value=16 over the 0..32 range of ref_value=10 + 0.5*i quantises
    // to ≲ 0.5 resolution; the tolerance accounts for that plus the zero
    // substitution at NaN positions perturbing the range derivation.
    for i in [0, 10, 15, 25, 31] {
        assert!(
            (got[i] - values[i]).abs() <= 0.5,
            "pos {i}: {} vs {}",
            got[i],
            values[i]
        );
    }
}

// ── f64 × shuffle filter ───────────────────────────────────────────────────

#[test]
fn f64_shuffle_filter_with_nan_round_trip() {
    let values = vec![f64::NAN, 1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0, 7.0];
    let mut desc = make_descriptor(vec![8], Dtype::Float64);
    desc.filter = "shuffle".to_string();
    desc.params.insert(
        "shuffle_element_size".to_string(),
        ciborium::Value::Integer(8.into()),
    );

    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let data = f64_bytes(&values);
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();

    let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    assert_f64_matches(&objects[0].1, &values, "shuffle/nan");
}

// ── f64 × zstd compression ─────────────────────────────────────────────────

#[cfg(feature = "zstd")]
#[test]
fn f64_zstd_compression_with_nan_round_trip() {
    let mut values: Vec<f64> = (0..64).map(|i| i as f64).collect();
    for idx in [7, 42, 50] {
        values[idx] = f64::NAN;
    }
    let mut desc = make_descriptor(vec![64], Dtype::Float64);
    desc.compression = "zstd".to_string();
    desc.params
        .insert("zstd_level".to_string(), ciborium::Value::Integer(3.into()));
    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let data = f64_bytes(&values);
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();
    let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    assert_f64_matches(&objects[0].1, &values, "zstd/nan");
}

// ── f64 × blosc2 compression ───────────────────────────────────────────────

#[cfg(feature = "blosc2")]
#[test]
fn f64_blosc2_compression_with_nan_round_trip() {
    let mut values: Vec<f64> = (0..64).map(|i| i as f64 * 0.1).collect();
    values[10] = f64::INFINITY;
    values[30] = f64::NEG_INFINITY;
    let mut desc = make_descriptor(vec![64], Dtype::Float64);
    desc.compression = "blosc2".to_string();
    desc.params.insert(
        "blosc2_codec".to_string(),
        ciborium::Value::Text("lz4".to_string()),
    );
    desc.params.insert(
        "blosc2_clevel".to_string(),
        ciborium::Value::Integer(5.into()),
    );
    let options = EncodeOptions {
        allow_inf: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let data = f64_bytes(&values);
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();
    let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    assert_f64_matches(&objects[0].1, &values, "blosc2/inf");
}

// ── c64 exhaustive matrix ─────────────────────────────────────────────────

fn c64_bytes(values: &[(f32, f32)]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|(r, i)| {
            let mut b = Vec::with_capacity(8);
            b.extend_from_slice(&r.to_ne_bytes());
            b.extend_from_slice(&i.to_ne_bytes());
            b
        })
        .collect()
}

/// c64 input-shape variants — priority rule in effect: NaN beats Inf,
/// +Inf beats -Inf.  Each element is a (real, imag) pair.
fn c64_input_shapes() -> Vec<(&'static str, Vec<(f32, f32)>)> {
    vec![
        (
            "no-nonfinite",
            vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)],
        ),
        (
            "nan-only",
            vec![(f32::NAN, 1.0), (2.0, f32::NAN), (3.0, 3.0), (4.0, 4.0)],
        ),
        (
            "pos_inf-only",
            vec![
                (f32::INFINITY, 1.0),
                (2.0, f32::INFINITY),
                (3.0, 3.0),
                (4.0, 4.0),
            ],
        ),
        (
            "neg_inf-only",
            vec![
                (f32::NEG_INFINITY, 1.0),
                (2.0, f32::NEG_INFINITY),
                (3.0, 3.0),
                (4.0, 4.0),
            ],
        ),
        (
            "all-three-kinds",
            vec![
                (f32::NAN, 0.0),
                (f32::INFINITY, 0.0),
                (f32::NEG_INFINITY, 0.0),
                (3.0, 3.0),
            ],
        ),
    ]
}

/// Verify c64 output — NaN / Inf positions restored canonically to BOTH
/// components (documented lossy behaviour, §7.1).
fn assert_c64_matches(got_bytes: &[u8], original: &[(f32, f32)], label: &str) {
    let got: Vec<(f32, f32)> = got_bytes
        .chunks_exact(8)
        .map(|c| {
            let r = f32::from_ne_bytes([c[0], c[1], c[2], c[3]]);
            let i = f32::from_ne_bytes([c[4], c[5], c[6], c[7]]);
            (r, i)
        })
        .collect();
    assert_eq!(got.len(), original.len(), "[{label}]");
    for (idx, ((gr, gi), (r, i))) in got.iter().zip(original.iter()).enumerate() {
        let has_nan = r.is_nan() || i.is_nan();
        let has_pos_inf = !has_nan
            && ((r.is_infinite() && r.is_sign_positive())
                || (i.is_infinite() && i.is_sign_positive()));
        let has_neg_inf = !has_nan
            && !has_pos_inf
            && ((r.is_infinite() && r.is_sign_negative())
                || (i.is_infinite() && i.is_sign_negative()));
        if has_nan {
            assert!(
                gr.is_nan() && gi.is_nan(),
                "[{label}] pos {idx}: expected NaN pair"
            );
        } else if has_pos_inf {
            assert!(
                gr.is_infinite()
                    && gr.is_sign_positive()
                    && gi.is_infinite()
                    && gi.is_sign_positive(),
                "[{label}] pos {idx}: expected +Inf pair"
            );
        } else if has_neg_inf {
            assert!(
                gr.is_infinite()
                    && gr.is_sign_negative()
                    && gi.is_infinite()
                    && gi.is_sign_negative(),
                "[{label}] pos {idx}: expected -Inf pair"
            );
        } else {
            assert_eq!(*gr, *r, "[{label}] pos {idx} real");
            assert_eq!(*gi, *i, "[{label}] pos {idx} imag");
        }
    }
}

#[test]
fn c64_exhaustive_encoding_none_filter_none_compression_none() {
    for (shape_name, values) in c64_input_shapes() {
        for (method_name, method) in all_mask_methods() {
            let label = format!("c64/none/{shape_name}/nan_method={method_name}");
            let desc = make_descriptor(vec![values.len() as u64], Dtype::Complex64);
            let options = EncodeOptions {
                allow_nan: true,
                allow_inf: true,
                hash_algorithm: None,
                nan_mask_method: method.clone(),
                pos_inf_mask_method: method.clone(),
                neg_inf_mask_method: method.clone(),
                small_mask_threshold_bytes: 0,
                ..Default::default()
            };
            let data = c64_bytes(&values);
            let msg = encode(&make_global_meta(), &[(&desc, &data)], &options)
                .unwrap_or_else(|e| panic!("[{label}] encode: {e}"));

            // decode path
            let (_, objects) = decode(&msg, &DecodeOptions::default())
                .unwrap_or_else(|e| panic!("[{label}] decode: {e}"));
            assert_c64_matches(&objects[0].1, &values, &label);

            // decode_range(single) on whole array
            let (_, parts) = decode_range(
                &msg,
                0,
                &[(0, values.len() as u64)],
                &DecodeOptions::default(),
            )
            .unwrap();
            assert_c64_matches(&parts[0], &values, &format!("{label}/range"));
        }
    }
}

// ── Sampled f32 / f16 / bf16 / c128 ────────────────────────────────────────

#[test]
fn f32_with_nan_round_trip_sampled() {
    let values: Vec<f32> = vec![1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 5.0];
    let desc = make_descriptor(vec![5], Dtype::Float32);
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let options = EncodeOptions {
        allow_nan: true,
        allow_inf: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();
    let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    let got: Vec<f32> = objects[0]
        .1
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got[0], 1.0);
    assert!(got[1].is_nan());
    assert!(got[2].is_infinite() && got[2].is_sign_positive());
    assert!(got[3].is_infinite() && got[3].is_sign_negative());
    assert_eq!(got[4], 5.0);
}

#[test]
fn f16_with_nan_round_trip_sampled() {
    // float16: 0x7E00=NaN, 0x7C00=+Inf, 0xFC00=-Inf, 0x3C00=1.0
    let bits: Vec<u16> = vec![0x3C00, 0x7E00, 0x7C00, 0xFC00, 0x4000]; // 1.0, NaN, +Inf, -Inf, 2.0
    let desc = make_descriptor(vec![bits.len() as u64], Dtype::Float16);
    let data: Vec<u8> = bits.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let options = EncodeOptions {
        allow_nan: true,
        allow_inf: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();
    let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    let got: Vec<u16> = objects[0]
        .1
        .chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(got[0], 0x3C00); // 1.0 finite
    // canonical f16 NaN patterns restored at masked positions
    assert!(got[1] & 0x7C00 == 0x7C00 && got[1] & 0x03FF != 0); // NaN
    assert_eq!(got[2] & 0x7FFF, 0x7C00); // +Inf
    assert_eq!(got[3], 0xFC00); // -Inf
    assert_eq!(got[4], 0x4000); // 2.0 finite
}

#[test]
fn bf16_with_nan_round_trip_sampled() {
    // bf16: 0x7FC0=NaN, 0x7F80=+Inf, 0xFF80=-Inf, 0x3F80=1.0
    let bits: Vec<u16> = vec![0x3F80, 0x7FC0, 0x7F80, 0xFF80];
    let desc = make_descriptor(vec![bits.len() as u64], Dtype::Bfloat16);
    let data: Vec<u8> = bits.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let options = EncodeOptions {
        allow_nan: true,
        allow_inf: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();
    let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    let got: Vec<u16> = objects[0]
        .1
        .chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(got[0], 0x3F80); // 1.0
    assert!(got[1] & 0x7F80 == 0x7F80 && got[1] & 0x007F != 0); // NaN
    assert_eq!(got[2], 0x7F80); // +Inf
    assert_eq!(got[3], 0xFF80); // -Inf
}

#[test]
fn c128_with_nan_round_trip_sampled() {
    // Two complex128 elements: (1+2i) and (NaN+Inf i) — NaN dominates
    let data: Vec<u8> = [1.0_f64, 2.0, f64::NAN, f64::INFINITY]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    let desc = make_descriptor(vec![2], Dtype::Complex128);
    let options = EncodeOptions {
        allow_nan: true,
        allow_inf: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();
    let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    let got: Vec<f64> = objects[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got[0], 1.0);
    assert_eq!(got[1], 2.0);
    // Element 1 restored as (NaN, NaN) — both components canonical NaN
    assert!(got[2].is_nan());
    assert!(got[3].is_nan());
}
