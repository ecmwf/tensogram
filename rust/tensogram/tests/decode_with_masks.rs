// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Integration tests for decode-side NaN / Inf reconstruction and
//! the `decode_with_masks` API.  See
//! `docs/src/guide/nan-inf-handling.md` for the user contract and
//! `plans/WIRE_FORMAT.md` §6.5 for the wire-format spec.
//!
//! These cover:
//! - Default decode (`restore_non_finite = true`) restores canonical
//!   NaN / ±Inf at masked positions.
//! - Opt-out (`restore_non_finite = false`) returns the substituted
//!   zeros as they are on disk.
//! - `decode_with_masks` returns 0-substituted payload AND raw
//!   decompressed bitmasks for advanced callers.
//! - Every [`MaskMethod`] round-trips correctly.
//! - Hash verification still passes with masks present.
//! - Large (threshold-exceeding) masks via every method.

use std::collections::BTreeMap;
use tensogram::encode::MaskMethod;
use tensogram::*;

fn make_global_meta() -> GlobalMetadata {
    GlobalMetadata {
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

// ── decode_with_masks ───────────────────────────────────────────────────────

#[test]
fn decode_with_masks_returns_substituted_payload_plus_masks() {
    let data = f64_bytes(&[1.0, f64::NAN, 2.0, f64::INFINITY, 3.0]);
    let desc = make_descriptor(vec![5], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        allow_inf: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();

    let (_, objects) = decode_with_masks(&msg, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
    let obj = &objects[0];

    // Payload is 0-substituted regardless of restore_non_finite setting.
    let expected = f64_bytes(&[1.0, 0.0, 2.0, 0.0, 3.0]);
    assert_eq!(obj.payload, expected);

    // Both kinds of masks are present, with correct bit positions.
    assert_eq!(
        obj.masks.nan.as_ref().unwrap(),
        &vec![false, true, false, false, false]
    );
    assert_eq!(
        obj.masks.pos_inf.as_ref().unwrap(),
        &vec![false, false, false, true, false]
    );
    assert!(obj.masks.neg_inf.is_none());
}

#[test]
fn decode_with_masks_empty_when_no_masks_present() {
    let data = f64_bytes(&[1.0, 2.0, 3.0]);
    let desc = make_descriptor(vec![3], Dtype::Float64);
    let options = EncodeOptions {
        hash_algorithm: None,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();

    let (_, objects) = decode_with_masks(&msg, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
    let obj = &objects[0];
    assert!(obj.masks.is_empty());
    assert!(obj.descriptor.masks.is_none());
    assert_eq!(obj.payload, f64_bytes(&[1.0, 2.0, 3.0]));
}

// ── Mask-method round-trips ─────────────────────────────────────────────────

fn round_trip_with_method(method: MaskMethod) {
    // Build a large enough payload that every method runs (not
    // auto-forced to 'none').  128 elements × 8 bytes = 1024 byte
    // payload; the packed mask is 128/8 = 16 bytes — set threshold
    // to 0 so small-mask fallback doesn't kick in either way.
    let mut values: Vec<f64> = (0..128).map(|i| i as f64).collect();
    for idx in [10usize, 50, 100] {
        values[idx] = f64::NAN;
    }
    let data = f64_bytes(&values);
    let desc = make_descriptor(vec![128], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        nan_mask_method: method.clone(),
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
    for (i, v) in got.iter().enumerate() {
        if [10, 50, 100].contains(&i) {
            assert!(v.is_nan(), "method {method:?}: pos {i} should be NaN");
        } else {
            assert_eq!(*v, i as f64, "method {method:?}: finite mismatch at {i}");
        }
    }
}

#[test]
fn round_trip_method_none() {
    round_trip_with_method(MaskMethod::None);
}

#[test]
fn round_trip_method_rle() {
    round_trip_with_method(MaskMethod::Rle);
}

#[test]
fn round_trip_method_roaring() {
    round_trip_with_method(MaskMethod::Roaring);
}

#[cfg(feature = "lz4")]
#[test]
fn round_trip_method_lz4() {
    round_trip_with_method(MaskMethod::Lz4);
}

#[cfg(feature = "zstd")]
#[test]
fn round_trip_method_zstd_default_level() {
    round_trip_with_method(MaskMethod::Zstd { level: None });
}

#[cfg(feature = "zstd")]
#[test]
fn round_trip_method_zstd_explicit_level() {
    round_trip_with_method(MaskMethod::Zstd { level: Some(19) });
}

#[cfg(feature = "blosc2")]
#[test]
fn round_trip_method_blosc2() {
    use tensogram_encodings::pipeline::Blosc2Codec;
    round_trip_with_method(MaskMethod::Blosc2 {
        codec: Blosc2Codec::Lz4,
        level: 5,
    });
}

// ── Restore off / on compare ────────────────────────────────────────────────

#[test]
fn restore_non_finite_false_keeps_zeros() {
    let data = f64_bytes(&[0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0]);
    let desc = make_descriptor(vec![5], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        allow_inf: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();

    let opts_off = DecodeOptions {
        restore_non_finite: false,
        ..Default::default()
    };
    let (_, objects) = decode(&msg, &opts_off).unwrap();
    let expected = f64_bytes(&[0.0, 0.0, 0.0, 0.0, 1.0]);
    assert_eq!(objects[0].1, expected);
}

// ── Large payloads ──────────────────────────────────────────────────────────

#[test]
fn large_sparse_nan_payload_round_trips() {
    let n = 10_000;
    let mut values: Vec<f64> = (0..n).map(|i| i as f64).collect();
    // Scatter ~1% NaNs.
    for i in (17..n).step_by(97) {
        values[i] = f64::NAN;
    }
    let data = f64_bytes(&values);
    let desc = make_descriptor(vec![n as u64], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: Some(HashAlgorithm::Xxh3),
        // small_mask_threshold stays at default 128; 10_000 bits is
        // ~1_250 bytes so roaring is used.
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();

    let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    let got: Vec<f64> = objects[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    for (i, &v) in got.iter().enumerate() {
        let expected_nan = (i >= 17) && ((i - 17) % 97 == 0);
        if expected_nan {
            assert!(v.is_nan(), "pos {i} should be NaN");
        } else {
            assert_eq!(v, i as f64);
        }
    }
}

// ── decode_object path respects restore_non_finite ──────────────────────────

#[test]
fn decode_object_restores_nan() {
    let data = f64_bytes(&[1.0, f64::NAN, 3.0]);
    let desc = make_descriptor(vec![3], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();

    let (_, _, decoded) = decode_object(&msg, 0, &DecodeOptions::default()).unwrap();
    let got: Vec<f64> = decoded
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got[0], 1.0);
    assert!(got[1].is_nan());
    assert_eq!(got[2], 3.0);
}
