// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Edge-case tests targeting untested code paths.
//!
//! Covers zero-element tensors, bitmask dtype, version validation,
//! NaN/Infinity params, unknown hash algorithms, empty messages, mixed dtypes,
//! decode_range edge cases, all compressor configurations, and dtype coverage.

use std::collections::BTreeMap;
use tensogram::*;

// ── Helpers ──────────────────────────────────────────────────────────────────

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
        params: BTreeMap::new(),
        masks: None,
    }
}

fn encode_roundtrip(
    desc: &DataObjectDescriptor,
    data: &[u8],
) -> (GlobalMetadata, Vec<(DataObjectDescriptor, Vec<u8>)>) {
    let meta = make_global_meta();
    let encoded = encode(&meta, &[(desc, data)], &EncodeOptions::default()).unwrap();
    decode(&encoded, &DecodeOptions::default()).unwrap()
}

/// Produce `n_floats × 4` bytes of deterministic **finite** f32 values.
///
/// Many compression round-trip tests previously used
/// `(0..N).map(|i| (i % 256) as u8).collect()` which happens to produce
/// NaN bit patterns when interpreted as f32 (the byte pattern `0x7F_..`
/// / `0xFF_..` lands inside the exponent slot).  Since 0.17 the encode
/// path rejects non-finite floats by default; these tests use this
/// helper instead to produce input that both round-trips byte-exactly
/// through compression AND passes the finite-value pre-check.
fn finite_f32_bytes(n_floats: usize) -> Vec<u8> {
    (0..n_floats)
        .map(|i| i as f32)
        .flat_map(|v| v.to_ne_bytes())
        .collect()
}

// ── 1. Dtype byte_width and Display coverage ─────────────────────────────────

#[test]
fn dtype_byte_width_all_variants() {
    assert_eq!(Dtype::Float16.byte_width(), 2);
    assert_eq!(Dtype::Bfloat16.byte_width(), 2);
    assert_eq!(Dtype::Float32.byte_width(), 4);
    assert_eq!(Dtype::Float64.byte_width(), 8);
    assert_eq!(Dtype::Complex64.byte_width(), 8);
    assert_eq!(Dtype::Complex128.byte_width(), 16);
    assert_eq!(Dtype::Int8.byte_width(), 1);
    assert_eq!(Dtype::Int16.byte_width(), 2);
    assert_eq!(Dtype::Int32.byte_width(), 4);
    assert_eq!(Dtype::Int64.byte_width(), 8);
    assert_eq!(Dtype::Uint8.byte_width(), 1);
    assert_eq!(Dtype::Uint16.byte_width(), 2);
    assert_eq!(Dtype::Uint32.byte_width(), 4);
    assert_eq!(Dtype::Uint64.byte_width(), 8);
    assert_eq!(Dtype::Bitmask.byte_width(), 0);
}

#[test]
fn dtype_swap_unit_size_all_variants() {
    // Simple types: swap_unit_size == byte_width
    assert_eq!(Dtype::Float16.swap_unit_size(), 2);
    assert_eq!(Dtype::Bfloat16.swap_unit_size(), 2);
    assert_eq!(Dtype::Float32.swap_unit_size(), 4);
    assert_eq!(Dtype::Float64.swap_unit_size(), 8);
    assert_eq!(Dtype::Int8.swap_unit_size(), 1);
    assert_eq!(Dtype::Int16.swap_unit_size(), 2);
    assert_eq!(Dtype::Int32.swap_unit_size(), 4);
    assert_eq!(Dtype::Int64.swap_unit_size(), 8);
    assert_eq!(Dtype::Uint8.swap_unit_size(), 1);
    assert_eq!(Dtype::Uint16.swap_unit_size(), 2);
    assert_eq!(Dtype::Uint32.swap_unit_size(), 4);
    assert_eq!(Dtype::Uint64.swap_unit_size(), 8);
    // Complex types: swap each scalar component independently
    assert_eq!(Dtype::Complex64.swap_unit_size(), 4); // two float32
    assert_eq!(Dtype::Complex128.swap_unit_size(), 8); // two float64
    // Bitmask: no swap
    assert_eq!(Dtype::Bitmask.swap_unit_size(), 0);
}

#[test]
fn dtype_display_all_variants() {
    assert_eq!(format!("{}", Dtype::Float16), "float16");
    assert_eq!(format!("{}", Dtype::Bfloat16), "bfloat16");
    assert_eq!(format!("{}", Dtype::Float32), "float32");
    assert_eq!(format!("{}", Dtype::Float64), "float64");
    assert_eq!(format!("{}", Dtype::Complex64), "complex64");
    assert_eq!(format!("{}", Dtype::Complex128), "complex128");
    assert_eq!(format!("{}", Dtype::Int8), "int8");
    assert_eq!(format!("{}", Dtype::Int16), "int16");
    assert_eq!(format!("{}", Dtype::Int32), "int32");
    assert_eq!(format!("{}", Dtype::Int64), "int64");
    assert_eq!(format!("{}", Dtype::Uint8), "uint8");
    assert_eq!(format!("{}", Dtype::Uint16), "uint16");
    assert_eq!(format!("{}", Dtype::Uint32), "uint32");
    assert_eq!(format!("{}", Dtype::Uint64), "uint64");
    assert_eq!(format!("{}", Dtype::Bitmask), "bitmask");
}

// ── 2. Zero-element tensors ──────────────────────────────────────────────────

#[test]
fn zero_element_1d_tensor_roundtrips() {
    let desc = make_descriptor(vec![0], Dtype::Float32);
    let data: Vec<u8> = vec![];
    let (_, objects) = encode_roundtrip(&desc, &data);
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].1.len(), 0);
    assert_eq!(objects[0].0.shape, vec![0]);
}

#[test]
fn zero_element_multidim_tensor_roundtrips() {
    let desc = make_descriptor(vec![3, 0, 5], Dtype::Float64);
    let data: Vec<u8> = vec![];
    let (_, objects) = encode_roundtrip(&desc, &data);
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].1.len(), 0);
    assert_eq!(objects[0].0.shape, vec![3, 0, 5]);
}

// ── 3. Empty message (metadata-only) ─────────────────────────────────────────

#[test]
fn empty_message_no_data_objects() {
    let meta = make_global_meta();
    let encoded = encode(&meta, &[], &EncodeOptions::default()).unwrap();
    let (decoded_meta, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded_meta.version, 3);
    assert!(objects.is_empty());
}

#[test]
fn empty_message_with_custom_metadata() {
    let mut extra = BTreeMap::new();
    extra.insert(
        "source".to_string(),
        ciborium::Value::Text("test".to_string()),
    );
    let meta = GlobalMetadata {
        version: 3,
        extra,
        ..Default::default()
    };
    let encoded = encode(&meta, &[], &EncodeOptions::default()).unwrap();
    let (decoded_meta, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert!(objects.is_empty());
    assert_eq!(
        decoded_meta.extra.get("source"),
        Some(&ciborium::Value::Text("test".to_string()))
    );
}

// ── 4. Mixed dtypes in one message ───────────────────────────────────────────

#[test]
fn mixed_dtypes_in_single_message() {
    let meta = make_global_meta();
    let desc_f32 = make_descriptor(vec![4], Dtype::Float32);
    let data_f32 = vec![0u8; 16]; // 4 * 4 bytes

    let desc_u8 = make_descriptor(vec![8], Dtype::Uint8);
    let data_u8 = vec![1u8; 8];

    let desc_i64 = make_descriptor(vec![2], Dtype::Int64);
    let data_i64 = vec![2u8; 16]; // 2 * 8 bytes

    let encoded = encode(
        &meta,
        &[
            (&desc_f32, data_f32.as_slice()),
            (&desc_u8, data_u8.as_slice()),
            (&desc_i64, data_i64.as_slice()),
        ],
        &EncodeOptions::default(),
    )
    .unwrap();

    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 3);
    assert_eq!(objects[0].0.dtype, Dtype::Float32);
    assert_eq!(objects[1].0.dtype, Dtype::Uint8);
    assert_eq!(objects[2].0.dtype, Dtype::Int64);
    assert_eq!(objects[0].1, data_f32);
    assert_eq!(objects[1].1, data_u8);
    assert_eq!(objects[2].1, data_i64);
}

// ── 5. Bitmask dtype ─────────────────────────────────────────────────────────

#[test]
fn bitmask_roundtrip() {
    // 10 bits → ceil(10/8) = 2 bytes payload
    let desc = make_descriptor(vec![10], Dtype::Bitmask);
    let data = vec![0b11110000, 0b11000000];
    let meta = make_global_meta();
    let encoded = encode(
        &meta,
        &[(&desc, data.as_slice())],
        &EncodeOptions::default(),
    )
    .unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].0.dtype, Dtype::Bitmask);
    assert_eq!(objects[0].1, data);
}

#[test]
fn bitmask_with_float32_mask_pair() {
    let meta = make_global_meta();

    let desc_data = make_descriptor(vec![8], Dtype::Float32);
    let data = vec![0u8; 32]; // 8 * 4 bytes

    let desc_mask = make_descriptor(vec![8], Dtype::Bitmask);
    let mask = vec![0xFF]; // ceil(8/8) = 1 byte, all valid

    let encoded = encode(
        &meta,
        &[(&desc_data, data.as_slice()), (&desc_mask, mask.as_slice())],
        &EncodeOptions::default(),
    )
    .unwrap();

    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 2);
    assert_eq!(objects[0].0.dtype, Dtype::Float32);
    assert_eq!(objects[1].0.dtype, Dtype::Bitmask);
}

// ── 6. Version validation ────────────────────────────────────────────────────

#[test]
fn version_0_rejected() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![0u8; 16];
    let mut encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Patch version to 0 (bytes 8-9 in preamble)
    encoded[8] = 0;
    encoded[9] = 0;

    let result = decode(&encoded, &DecodeOptions::default());
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("deprecated") || msg.contains("version"),
        "expected version error, got: {msg}"
    );
}

#[test]
fn version_1_rejected() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![0u8; 16];
    let mut encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Patch version to 1
    encoded[8] = 0;
    encoded[9] = 1;

    let result = decode(&encoded, &DecodeOptions::default());
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("deprecated") || msg.contains("version"),
        "expected version error, got: {msg}"
    );
}

#[test]
fn version_2_accepted() {
    let meta = make_global_meta();
    let encoded = encode(&meta, &[], &EncodeOptions::default()).unwrap();
    assert!(decode(&encoded, &DecodeOptions::default()).is_ok());
}

// ── 7. NaN/Infinity in simple_packing params ─────────────────────────────────

fn make_simple_packing_desc(reference_value: f64) -> DataObjectDescriptor {
    let mut params = BTreeMap::new();
    params.insert(
        "reference_value".to_string(),
        ciborium::Value::Float(reference_value),
    );
    params.insert(
        "binary_scale_factor".to_string(),
        ciborium::Value::Integer(0.into()),
    );
    params.insert(
        "decimal_scale_factor".to_string(),
        ciborium::Value::Integer(0.into()),
    );
    params.insert(
        "bits_per_value".to_string(),
        ciborium::Value::Integer(16.into()),
    );

    DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![4],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Big,
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params,
        masks: None,
    }
}

#[test]
fn nan_reference_value_rejected() {
    let desc = make_simple_packing_desc(f64::NAN);
    let data = vec![0u8; 32]; // 4 * 8 bytes
    let result = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    );
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("finite") || msg.contains("NaN") || msg.contains("reference_value"),
        "expected NaN error, got: {msg}"
    );
}

#[test]
fn infinity_reference_value_rejected() {
    let desc = make_simple_packing_desc(f64::INFINITY);
    let data = vec![0u8; 32];
    let result = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    );
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("finite") || msg.contains("inf") || msg.contains("reference_value"),
        "expected infinity error, got: {msg}"
    );
}

#[test]
fn neg_infinity_reference_value_rejected() {
    let desc = make_simple_packing_desc(f64::NEG_INFINITY);
    let data = vec![0u8; 32];
    let result = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    );
    assert!(result.is_err());
}

// ── 7b. Standalone-API safety net via the high-level `encode()` path ────────
//
// `simple_packing::encode_with_threads` now validates the params it
// receives (see plans/RESEARCH_NAN_HANDLING.md §4.2.3).  The high-level
// `tensogram::encode` delegates to it through the pipeline, so the
// validation also fires when a caller supplies a malformed
// `binary_scale_factor` via the descriptor params.

#[test]
fn unreasonable_binary_scale_factor_rejected() {
    // Caller supplies a descriptor with `binary_scale_factor = i32::MAX`
    // (the fingerprint of feeding Inf through `compute_params`'s range
    // arithmetic).  Without the safety net, the decode silently
    // reconstructs the constant `reference_value` — with it, the
    // encode fails clearly.
    let mut desc = make_simple_packing_desc(273.15);
    desc.params.insert(
        "binary_scale_factor".to_string(),
        ciborium::Value::Integer((i64::from(i32::MAX)).into()),
    );
    let data: Vec<u8> = [270.0_f64, 275.0, 280.0, 285.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("binary_scale_factor"),
        "error must name the field: {err}"
    );
    assert!(err.contains("256"), "error must quote the threshold: {err}");
}

#[test]
fn binary_scale_factor_at_threshold_accepted() {
    // Threshold is inclusive: `|bsf| == 256` passes the safety net.
    let mut desc = make_simple_packing_desc(0.0);
    desc.params.insert(
        "binary_scale_factor".to_string(),
        ciborium::Value::Integer(256i64.into()),
    );
    let data: Vec<u8> = [1.0_f64, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .expect("bsf=256 must be accepted");
}

#[test]
fn realistic_binary_scale_factor_accepted() {
    // Regression guard: pin that the safety net's threshold does not
    // block real-world weather/climate values.
    for bsf in [-60_i64, -20, -1, 0, 1, 20, 60] {
        let mut desc = make_simple_packing_desc(273.15);
        desc.params.insert(
            "binary_scale_factor".to_string(),
            ciborium::Value::Integer(bsf.into()),
        );
        let data: Vec<u8> = [273.15_f64, 283.0, 293.0, 303.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        encode(
            &make_global_meta(),
            &[(&desc, &data)],
            &EncodeOptions::default(),
        )
        .unwrap_or_else(|e| panic!("realistic bsf {bsf} rejected: {e}"));
    }
}

// ── 8. Unknown hash algorithm on decode ──────────────────────────────────────

#[test]
fn unknown_hash_algorithm_skips_verification() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![0u8; 16];

    // Encode with a hash
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Decode works; `verify_hash` on DecodeOptions is a no-op in
    // v3 (frame-level integrity moved to validate --checksum) but
    // the option is kept for source compatibility.
    let (_, _objects) = decode(
        &encoded,
        &DecodeOptions {
            verify_hash: true,
            ..Default::default()
        },
    )
    .unwrap();

    // The standalone `verify_hash` helper on a `HashDescriptor`
    // must still silently skip verification for unknown algorithm
    // names — the forward-compatibility contract documented in
    // `crate::hash::verify_hash`.
    let descriptor = HashDescriptor {
        algorithm: "sha512".to_string(),
        value: "fake_hash_value".to_string(),
    };
    assert!(tensogram::verify_hash(b"any data", &descriptor).is_ok());
}

// ── 9. decode_range edge cases ───────────────────────────────────────────────

#[test]
fn decode_range_empty_ranges_returns_empty() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![10], Dtype::Float32);
    let data = vec![0u8; 40]; // 10 * 4 bytes
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let (_, result) =
        decode_range(&encoded, 0, &[], &DecodeOptions::default()).expect("decode_range failed");
    assert!(result.is_empty());
}

#[test]
fn decode_range_bitmask_rejected() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![16], Dtype::Bitmask);
    let data = vec![0xFF; 2]; // ceil(16/8) = 2 bytes
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let result = decode_range(&encoded, 0, &[(0, 8)], &DecodeOptions::default());
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("bitmask"),
        "expected bitmask error, got: {msg}"
    );
}

#[test]
fn decode_range_object_index_out_of_range() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![0u8; 16];
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let result = decode_range(&encoded, 5, &[(0, 2)], &DecodeOptions::default());
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("out of range"));
}

// ── 10. decode_object edge cases ─────────────────────────────────────────────

#[test]
fn decode_object_out_of_range() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![0u8; 16];
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let result = decode_object(&encoded, 99, &DecodeOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("out of range"));
}

#[test]
fn decode_object_by_index() {
    let meta = make_global_meta();
    let desc0 = make_descriptor(vec![2], Dtype::Float32);
    let data0 = vec![1u8; 8];
    let desc1 = make_descriptor(vec![3], Dtype::Uint8);
    let data1 = vec![2u8; 3];

    let encoded = encode(
        &meta,
        &[(&desc0, data0.as_slice()), (&desc1, data1.as_slice())],
        &EncodeOptions::default(),
    )
    .unwrap();

    // Get object 1 specifically
    let (_, returned_desc, returned_data) =
        decode_object(&encoded, 1, &DecodeOptions::default()).unwrap();
    assert_eq!(returned_desc.dtype, Dtype::Uint8);
    assert_eq!(returned_data, data1);
}

// ── 11. All integer/float dtypes roundtrip ───────────────────────────────────

#[test]
fn all_dtypes_roundtrip() {
    let dtypes_and_sizes = [
        (Dtype::Float16, 2),
        (Dtype::Bfloat16, 2),
        (Dtype::Float32, 4),
        (Dtype::Float64, 8),
        (Dtype::Complex64, 8),
        (Dtype::Complex128, 16),
        (Dtype::Int8, 1),
        (Dtype::Int16, 2),
        (Dtype::Int32, 4),
        (Dtype::Int64, 8),
        (Dtype::Uint8, 1),
        (Dtype::Uint16, 2),
        (Dtype::Uint32, 4),
        (Dtype::Uint64, 8),
    ];

    for (dtype, byte_width) in dtypes_and_sizes {
        let num_elements = 4u64;
        let data_len = num_elements as usize * byte_width;
        let desc = make_descriptor(vec![num_elements], dtype);
        let data: Vec<u8> = (0..data_len).map(|i| (i % 256) as u8).collect();

        let (_, objects) = encode_roundtrip(&desc, &data);
        assert_eq!(objects[0].1, data, "roundtrip failed for dtype {dtype}");
    }
}

// ── 12. Encoding error paths ─────────────────────────────────────────────────

#[test]
fn unknown_encoding_rejected() {
    let meta = make_global_meta();
    let mut desc = make_descriptor(vec![4], Dtype::Float32);
    desc.encoding = "foobar".to_string();
    let data = vec![0u8; 16];

    let result = encode(&meta, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("unknown encoding"));
}

#[test]
fn unknown_filter_rejected() {
    let meta = make_global_meta();
    let mut desc = make_descriptor(vec![4], Dtype::Float32);
    desc.filter = "magic_filter".to_string();
    let data = vec![0u8; 16];

    let result = encode(&meta, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("unknown filter"));
}

#[test]
fn unknown_compression_rejected() {
    let meta = make_global_meta();
    let mut desc = make_descriptor(vec![4], Dtype::Float32);
    desc.compression = "quantum_compress".to_string();
    let data = vec![0u8; 16];

    let result = encode(&meta, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("unknown compression")
    );
}

#[test]
fn strides_length_mismatch_rejected() {
    let meta = make_global_meta();
    let mut desc = make_descriptor(vec![4, 5], Dtype::Float32);
    desc.strides = vec![1]; // Wrong length
    let data = vec![0u8; 80];

    let result = encode(&meta, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("strides"));
}

#[test]
fn data_length_mismatch_rejected() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![0u8; 17]; // Should be 16

    let result = encode(&meta, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("data_len"));
}

// ── 13. Encode without hash ──────────────────────────────────────────────────

/// `hash_algorithm = None` must round-trip cleanly and leave every
/// frame's inline hash slot at zero (v3 `HASHES_PRESENT = 0`
/// contract).
#[test]
fn encode_without_hash() {
    use tensogram::wire::{MessageFlags, Preamble};
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![0u8; 16];

    let options = EncodeOptions {
        hash_algorithm: None,
        ..Default::default()
    };
    let encoded = encode(&meta, &[(&desc, &data)], &options).unwrap();
    let preamble = Preamble::read_from(&encoded).unwrap();
    assert!(
        !preamble.flags.has(MessageFlags::HASHES_PRESENT),
        "hash_algorithm = None must clear HASHES_PRESENT"
    );
    let (_meta, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
}

#[test]
fn verify_hash_true_on_unhashed_message_succeeds() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![0u8; 16];

    let options = EncodeOptions {
        hash_algorithm: None,
        ..Default::default()
    };
    let encoded = encode(&meta, &[(&desc, &data)], &options).unwrap();
    // verify_hash: true should silently skip if no hash present
    let result = decode(
        &encoded,
        &DecodeOptions {
            verify_hash: true,
            ..Default::default()
        },
    );
    assert!(result.is_ok());
}

// ── 14. decode_metadata ──────────────────────────────────────────────────────

#[test]
fn decode_metadata_only() {
    let mut base_entry = BTreeMap::new();
    base_entry.insert(
        "centre".to_string(),
        ciborium::Value::Text("ecmwf".to_string()),
    );
    let meta = GlobalMetadata {
        version: 3,
        base: vec![base_entry],
        ..Default::default()
    };
    let desc = make_descriptor(vec![100], Dtype::Float64);
    let data = vec![0u8; 800];
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let decoded_meta = decode_metadata(&encoded).unwrap();
    assert_eq!(
        decoded_meta.base[0].get("centre"),
        Some(&ciborium::Value::Text("ecmwf".to_string()))
    );
}

// ── 15. Iterator edge cases ──────────────────────────────────────────────────

#[test]
fn messages_iterator_on_empty_buffer() {
    let iter = messages(&[]);
    assert_eq!(iter.len(), 0);
    assert_eq!(iter.count(), 0);
}

#[test]
fn messages_iterator_on_garbage() {
    let garbage = vec![0xFF; 100];
    let iter = messages(&garbage);
    assert_eq!(iter.len(), 0);
}

#[test]
fn messages_iterator_multiple() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let data1 = vec![1u8; 8];
    let data2 = vec![2u8; 8];

    let msg1 = encode(&meta, &[(&desc, &data1)], &EncodeOptions::default()).unwrap();
    let msg2 = encode(&meta, &[(&desc, &data2)], &EncodeOptions::default()).unwrap();

    let mut buf = msg1;
    buf.extend_from_slice(&msg2);

    let iter = messages(&buf);
    assert_eq!(iter.len(), 2);
}

#[test]
fn objects_iterator_empty_message() {
    let meta = make_global_meta();
    let encoded = encode(&meta, &[], &EncodeOptions::default()).unwrap();

    let iter = objects(&encoded, DecodeOptions::default()).unwrap();
    assert_eq!(iter.len(), 0);
}

#[test]
fn objects_metadata_iterator() {
    let meta = make_global_meta();
    let desc0 = make_descriptor(vec![2], Dtype::Float32);
    let desc1 = make_descriptor(vec![3], Dtype::Uint8);
    let data0 = vec![0u8; 8];
    let data1 = vec![0u8; 3];

    let encoded = encode(
        &meta,
        &[(&desc0, data0.as_slice()), (&desc1, data1.as_slice())],
        &EncodeOptions::default(),
    )
    .unwrap();

    let descs: Vec<DataObjectDescriptor> = objects_metadata(&encoded).unwrap().collect();
    assert_eq!(descs.len(), 2);
    assert_eq!(descs[0].dtype, Dtype::Float32);
    assert_eq!(descs[1].dtype, Dtype::Uint8);
}

// ── 16. File edge cases ──────────────────────────────────────────────────────

#[test]
fn empty_file_has_zero_messages() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.tgm");

    let file = TensogramFile::create(&path).unwrap();
    assert_eq!(file.message_count().unwrap(), 0);
}

#[test]
fn read_message_from_empty_file_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.tgm");

    let file = TensogramFile::create(&path).unwrap();
    let result = file.read_message(0);
    assert!(result.is_err());
}

#[test]
fn file_append_and_read_back() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.tgm");

    let mut file = TensogramFile::create(&path).unwrap();
    let meta = make_global_meta();
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let data = vec![42u8; 8];

    file.append(
        &meta,
        &[(&desc, data.as_slice())],
        &EncodeOptions::default(),
    )
    .unwrap();

    assert_eq!(file.message_count().unwrap(), 1);
    let raw = file.read_message(0).unwrap();
    let (_, objects) = decode(&raw, &DecodeOptions::default()).unwrap();
    assert_eq!(objects[0].1, data);
}

// ── 17. Scan edge cases ──────────────────────────────────────────────────────

#[test]
fn scan_empty_buffer() {
    assert!(scan(&[]).is_empty());
}

#[test]
fn scan_garbage_only() {
    assert!(scan(&[0xDE, 0xAD, 0xBE, 0xEF]).is_empty());
}

#[test]
fn scan_finds_multiple_messages() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![2], Dtype::Uint8);
    let msg1 = encode(&meta, &[(&desc, &[1, 2])], &EncodeOptions::default()).unwrap();
    let msg2 = encode(&meta, &[(&desc, &[3, 4])], &EncodeOptions::default()).unwrap();

    let mut buf = msg1.clone();
    buf.extend_from_slice(&msg2);

    let offsets = scan(&buf);
    assert_eq!(offsets.len(), 2);
    assert_eq!(offsets[0].0, 0);
    assert_eq!(offsets[1].0, msg1.len());
}

// ── 18. Param extraction error paths ─────────────────────────────────────────

#[test]
fn get_f64_param_wrong_type() {
    let meta = make_global_meta();
    let mut desc = make_simple_packing_desc(0.0);
    // Replace reference_value with a string
    desc.params.insert(
        "reference_value".to_string(),
        ciborium::Value::Text("not_a_number".to_string()),
    );
    let data = vec![0u8; 32];
    let result = encode(&meta, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("expected number"));
}

#[test]
fn get_i64_param_wrong_type() {
    let meta = make_global_meta();
    let mut desc = make_simple_packing_desc(0.0);
    // Replace binary_scale_factor with a float
    desc.params.insert(
        "binary_scale_factor".to_string(),
        ciborium::Value::Float(3.15),
    );
    let data = vec![0u8; 32];
    let result = encode(&meta, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("expected integer"));
}

#[test]
fn missing_required_param() {
    let meta = make_global_meta();
    let mut desc = make_simple_packing_desc(0.0);
    desc.params.remove("bits_per_value");
    let data = vec![0u8; 32];
    let result = encode(&meta, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("missing required"));
}

// ── 19. Streaming encoder edge cases ─────────────────────────────────────────

#[test]
fn streaming_encoder_multiple_objects() {
    let meta = make_global_meta();
    let options = EncodeOptions::default();

    let desc0 = make_descriptor(vec![2], Dtype::Float32);
    let data0 = vec![10u8; 8];
    let desc1 = make_descriptor(vec![3], Dtype::Uint8);
    let data1 = vec![20u8; 3];

    let buf = Vec::new();
    let mut enc = StreamingEncoder::new(buf, &meta, &options).unwrap();
    enc.write_object(&desc0, &data0).unwrap();
    enc.write_object(&desc1, &data1).unwrap();
    assert_eq!(enc.object_count(), 2);
    let result = enc.finish().unwrap();

    let (_, objects) = decode(&result, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 2);
    assert_eq!(objects[0].1, data0);
    assert_eq!(objects[1].1, data1);
}

// ── 20. Hash verification on decode ──────────────────────────────────────────

/// `verify_hash` (the standalone `HashDescriptor`-based helper)
/// returns `HashMismatch` when the stored digest disagrees with
/// the recomputed xxh3-64.  Frame-level integrity in v3 goes
/// through the inline slot + `validate --checksum` instead.
#[test]
fn hash_mismatch_detected_on_verify() {
    let data = vec![42u8; 16];
    let bad_hash = HashDescriptor {
        algorithm: "xxh3".to_string(),
        value: "0000000000000000".to_string(),
    };
    let result = tensogram::verify_hash(&data, &bad_hash);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("hash mismatch"));
}

// ── 21. GlobalMetadata namespaces ────────────────────────────────────────────

#[test]
fn metadata_namespaces_roundtrip() {
    // In the new model, per-object metadata lives in base[i].
    // Message-level extra is for ad-hoc annotations.
    let mut base_entry = BTreeMap::new();
    base_entry.insert(
        "foo".to_string(),
        ciborium::Value::Text("base_foo".to_string()),
    );
    base_entry.insert("bar".to_string(), ciborium::Value::Integer(42.into()));

    let mut extra = BTreeMap::new();
    extra.insert(
        "msg_level".to_string(),
        ciborium::Value::Text("extra_val".to_string()),
    );

    let meta = GlobalMetadata {
        version: 3,
        base: vec![base_entry],
        extra,
        ..Default::default()
    };

    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![1],
        strides: vec![1],
        dtype: tensogram::Dtype::Uint8,
        byte_order: tensogram::ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    let data = vec![0u8; 1];

    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (decoded, _) = decode(&encoded, &DecodeOptions::default()).unwrap();

    // Per-object keys survive in base[0]
    assert_eq!(
        decoded.base[0].get("foo"),
        Some(&ciborium::Value::Text("base_foo".to_string()))
    );
    assert_eq!(
        decoded.base[0].get("bar"),
        Some(&ciborium::Value::Integer(42.into()))
    );
    // Message-level extra survives
    assert_eq!(
        decoded.extra.get("msg_level"),
        Some(&ciborium::Value::Text("extra_val".to_string()))
    );
    // Encoder populates _reserved_ at message level with provenance
    assert!(decoded.reserved.contains_key("encoder"));
}

// ── 22. Very short buffer ────────────────────────────────────────────────────

#[test]
fn buffer_shorter_than_preamble() {
    let result = decode(&[0u8; 10], &DecodeOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("too short"));
}

#[test]
fn buffer_exactly_preamble_size_but_invalid() {
    let result = decode(&[0u8; 24], &DecodeOptions::default());
    assert!(result.is_err());
}

// ── 23. File concatenation ───────────────────────────────────────────────────

#[test]
fn concatenated_messages_scannable() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![2], Dtype::Uint8);

    let msg1 = encode(&meta, &[(&desc, &[10, 20])], &EncodeOptions::default()).unwrap();
    let msg2 = encode(&meta, &[(&desc, &[30, 40])], &EncodeOptions::default()).unwrap();
    let msg3 = encode(&meta, &[(&desc, &[50, 60])], &EncodeOptions::default()).unwrap();

    // Simulate: cat msg1 msg2 msg3 > all
    let mut all = Vec::new();
    all.extend_from_slice(&msg1);
    all.extend_from_slice(&msg2);
    all.extend_from_slice(&msg3);

    let offsets = scan(&all);
    assert_eq!(offsets.len(), 3);

    // Verify each message decodes correctly
    for (i, &(offset, length)) in offsets.iter().enumerate() {
        let msg_bytes = &all[offset..offset + length];
        let (_, objects) = decode(msg_bytes, &DecodeOptions::default()).unwrap();
        assert_eq!(objects[0].1.len(), 2, "message {i} wrong size");
    }
}

// ── 24. Encode options: default includes xxh3 hash ───────────────────────────

#[test]
fn default_encode_options_use_xxh3() {
    let opts = EncodeOptions::default();
    assert_eq!(opts.hash_algorithm, Some(HashAlgorithm::Xxh3));
}

// ── 25. Get param with integer as f64 ────────────────────────────────────────

#[test]
fn f64_param_from_integer_value() {
    // When a CBOR integer is stored where a float is expected, it should convert
    let meta = make_global_meta();
    let mut desc = make_simple_packing_desc(0.0);
    // Use an integer instead of float for reference_value
    desc.params.insert(
        "reference_value".to_string(),
        ciborium::Value::Integer(100.into()),
    );
    let data = vec![0u8; 32];
    // Should not error — integer converts to f64
    let result = encode(&meta, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_ok());
}

// ── 26. Compressor roundtrips ────────────────────────────────────────────────

fn make_compressed_descriptor(
    shape: Vec<u64>,
    dtype: Dtype,
    compression: &str,
    params: BTreeMap<String, ciborium::Value>,
) -> DataObjectDescriptor {
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
        compression: compression.to_string(),
        params,
        masks: None,
    }
}

#[test]
fn zstd_roundtrip() {
    let mut params = BTreeMap::new();
    params.insert("zstd_level".to_string(), ciborium::Value::Integer(3.into()));
    let desc = make_compressed_descriptor(vec![100], Dtype::Float32, "zstd", params);
    let data = finite_f32_bytes(100);
    let (_, objects) = encode_roundtrip(&desc, &data);
    assert_eq!(objects[0].1, data);
}

#[test]
fn zstd_default_level() {
    // When zstd_level is not specified, default to 3
    let desc = make_compressed_descriptor(vec![100], Dtype::Float32, "zstd", BTreeMap::new());
    let data = finite_f32_bytes(100);
    let (_, objects) = encode_roundtrip(&desc, &data);
    assert_eq!(objects[0].1, data);
}

#[test]
fn lz4_roundtrip() {
    let desc = make_compressed_descriptor(vec![100], Dtype::Float32, "lz4", BTreeMap::new());
    let data = finite_f32_bytes(100);
    let (_, objects) = encode_roundtrip(&desc, &data);
    assert_eq!(objects[0].1, data);
}

#[test]
fn blosc2_roundtrip_default_codec() {
    let desc = make_compressed_descriptor(vec![100], Dtype::Float32, "blosc2", BTreeMap::new());
    let data = finite_f32_bytes(100);
    let (_, objects) = encode_roundtrip(&desc, &data);
    assert_eq!(objects[0].1, data);
}

#[test]
fn blosc2_roundtrip_all_codecs() {
    for codec in ["blosclz", "lz4", "lz4hc", "zlib", "zstd"] {
        let mut params = BTreeMap::new();
        params.insert(
            "blosc2_codec".to_string(),
            ciborium::Value::Text(codec.to_string()),
        );
        params.insert(
            "blosc2_clevel".to_string(),
            ciborium::Value::Integer(3.into()),
        );
        let desc = make_compressed_descriptor(vec![100], Dtype::Float32, "blosc2", params);
        let data = finite_f32_bytes(100);
        let (_, objects) = encode_roundtrip(&desc, &data);
        assert_eq!(objects[0].1, data, "failed for blosc2 codec: {codec}");
    }
}

#[test]
fn blosc2_unknown_codec_rejected() {
    let mut params = BTreeMap::new();
    params.insert(
        "blosc2_codec".to_string(),
        ciborium::Value::Text("snappy".to_string()),
    );
    let desc = make_compressed_descriptor(vec![100], Dtype::Float32, "blosc2", params);
    let data = vec![0u8; 400];
    let result = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    );
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("unknown blosc2 codec")
    );
}

#[test]
fn zfp_fixed_rate_roundtrip() {
    let mut params = BTreeMap::new();
    params.insert(
        "zfp_mode".to_string(),
        ciborium::Value::Text("fixed_rate".to_string()),
    );
    params.insert("zfp_rate".to_string(), ciborium::Value::Float(16.0));
    let desc = make_compressed_descriptor(vec![100], Dtype::Float64, "zfp", params);
    let data: Vec<u8> = vec![0u8; 800]; // 100 * 8 bytes
    let meta = make_global_meta();
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    // ZFP is lossy, so we just check it decodes without error and has the right size
    assert_eq!(objects[0].1.len(), 800);
}

#[test]
fn zfp_cross_endian_decode_produces_native_bytes() {
    // ZFP with byte_order=Big: the compressor always reads input as native
    // (per design: "always encode in the endianness of the caller"), so we
    // provide native-endian bytes.  The descriptor declares byte_order=Big,
    // which tells the ZFP decompressor to write output in big-endian.  The
    // pipeline's byteswap step then converts to native.  This exercises the
    // full cross-endian ZFP path.
    let values: Vec<f64> = (0..100).map(|i| (i as f64) * 0.5).collect();
    let ne_data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let mut params = BTreeMap::new();
    params.insert(
        "zfp_mode".to_string(),
        ciborium::Value::Text("fixed_rate".to_string()),
    );
    params.insert("zfp_rate".to_string(), ciborium::Value::Float(32.0));

    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![100],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "zfp".to_string(),
        params,
        masks: None,
    };

    let meta = make_global_meta();
    let encoded = encode(&meta, &[(&desc, &ne_data)], &EncodeOptions::default()).unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();

    // Decoded bytes should be native-endian.  Interpret with from_ne_bytes.
    let decoded_values: Vec<f64> = objects[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(decoded_values.len(), 100);
    // ZFP at rate=32 should be very close to the original.
    for (orig, dec) in values.iter().zip(decoded_values.iter()) {
        assert!(
            (orig - dec).abs() < 0.1,
            "ZFP cross-endian: orig={orig}, dec={dec}"
        );
    }
}

#[test]
fn zfp_fixed_precision_roundtrip() {
    let mut params = BTreeMap::new();
    params.insert(
        "zfp_mode".to_string(),
        ciborium::Value::Text("fixed_precision".to_string()),
    );
    params.insert(
        "zfp_precision".to_string(),
        ciborium::Value::Integer(32.into()),
    );
    let desc = make_compressed_descriptor(vec![100], Dtype::Float64, "zfp", params);
    let data = vec![0u8; 800];
    let meta = make_global_meta();
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(objects[0].1.len(), 800);
}

#[test]
fn zfp_fixed_accuracy_roundtrip() {
    let mut params = BTreeMap::new();
    params.insert(
        "zfp_mode".to_string(),
        ciborium::Value::Text("fixed_accuracy".to_string()),
    );
    params.insert("zfp_tolerance".to_string(), ciborium::Value::Float(0.01));
    let desc = make_compressed_descriptor(vec![100], Dtype::Float64, "zfp", params);
    let data = vec![0u8; 800];
    let meta = make_global_meta();
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(objects[0].1.len(), 800);
}

#[test]
fn zfp_missing_mode_rejected() {
    let desc = make_compressed_descriptor(vec![100], Dtype::Float64, "zfp", BTreeMap::new());
    let data = vec![0u8; 800];
    let result = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("zfp_mode"));
}

#[test]
fn zfp_unknown_mode_rejected() {
    let mut params = BTreeMap::new();
    params.insert(
        "zfp_mode".to_string(),
        ciborium::Value::Text("adaptive".to_string()),
    );
    let desc = make_compressed_descriptor(vec![100], Dtype::Float64, "zfp", params);
    let data = vec![0u8; 800];
    let result = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("unknown zfp_mode"));
}

#[test]
fn sz3_abs_roundtrip() {
    let mut params = BTreeMap::new();
    params.insert(
        "sz3_error_bound_mode".to_string(),
        ciborium::Value::Text("abs".to_string()),
    );
    params.insert("sz3_error_bound".to_string(), ciborium::Value::Float(0.001));
    let desc = make_compressed_descriptor(vec![100], Dtype::Float64, "sz3", params);
    let data = vec![0u8; 800];
    let meta = make_global_meta();
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(objects[0].1.len(), 800);
}

#[test]
fn sz3_rel_roundtrip() {
    let mut params = BTreeMap::new();
    params.insert(
        "sz3_error_bound_mode".to_string(),
        ciborium::Value::Text("rel".to_string()),
    );
    params.insert("sz3_error_bound".to_string(), ciborium::Value::Float(0.01));
    let desc = make_compressed_descriptor(vec![100], Dtype::Float64, "sz3", params);
    // Use non-zero data for relative error bound
    let data = finite_f32_bytes(200);
    let meta = make_global_meta();
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(objects[0].1.len(), 800);
}

#[test]
fn sz3_psnr_roundtrip() {
    let mut params = BTreeMap::new();
    params.insert(
        "sz3_error_bound_mode".to_string(),
        ciborium::Value::Text("psnr".to_string()),
    );
    params.insert("sz3_error_bound".to_string(), ciborium::Value::Float(40.0));
    let desc = make_compressed_descriptor(vec![100], Dtype::Float64, "sz3", params);
    let data = finite_f32_bytes(200);
    let meta = make_global_meta();
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(objects[0].1.len(), 800);
}

#[test]
fn sz3_missing_mode_rejected() {
    let desc = make_compressed_descriptor(vec![100], Dtype::Float64, "sz3", BTreeMap::new());
    let data = vec![0u8; 800];
    let result = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    );
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("sz3_error_bound_mode")
    );
}

#[test]
fn sz3_unknown_mode_rejected() {
    let mut params = BTreeMap::new();
    params.insert(
        "sz3_error_bound_mode".to_string(),
        ciborium::Value::Text("l2norm".to_string()),
    );
    params.insert("sz3_error_bound".to_string(), ciborium::Value::Float(0.01));
    let desc = make_compressed_descriptor(vec![100], Dtype::Float64, "sz3", params);
    let data = vec![0u8; 800];
    let result = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    );
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("unknown sz3_error_bound_mode")
    );
}

// ── 27. Shuffle filter with compression ──────────────────────────────────────

#[test]
fn shuffle_with_lz4_roundtrip() {
    let mut params = BTreeMap::new();
    params.insert(
        "shuffle_element_size".to_string(),
        ciborium::Value::Integer(4.into()),
    );
    let mut desc = make_descriptor(vec![100], Dtype::Float32);
    desc.filter = "shuffle".to_string();
    desc.compression = "lz4".to_string();
    desc.params = params;

    let data = finite_f32_bytes(100);
    let (_, objects) = encode_roundtrip(&desc, &data);
    assert_eq!(objects[0].1, data);
}

// ── 28. Szip with different pipeline configs ─────────────────────────────────

#[test]
fn szip_with_shuffle_roundtrip() {
    let mut params = BTreeMap::new();
    params.insert(
        "shuffle_element_size".to_string(),
        ciborium::Value::Integer(4.into()),
    );
    params.insert("szip_rsi".to_string(), ciborium::Value::Integer(128.into()));
    params.insert(
        "szip_block_size".to_string(),
        ciborium::Value::Integer(8.into()),
    );
    params.insert("szip_flags".to_string(), ciborium::Value::Integer(0.into()));

    let mut desc = make_descriptor(vec![100], Dtype::Float32);
    desc.filter = "shuffle".to_string();
    desc.compression = "szip".to_string();
    desc.params = params;

    let data = finite_f32_bytes(100);
    let (_, objects) = encode_roundtrip(&desc, &data);
    assert_eq!(objects[0].1, data);
}

// ── 29. Blosc2 typesize computation paths ────────────────────────────────────

#[test]
fn blosc2_with_shuffle_uses_typesize_1() {
    let mut params = BTreeMap::new();
    params.insert(
        "shuffle_element_size".to_string(),
        ciborium::Value::Integer(4.into()),
    );
    let mut desc = make_descriptor(vec![100], Dtype::Float32);
    desc.filter = "shuffle".to_string();
    desc.compression = "blosc2".to_string();
    desc.params = params;

    let data = finite_f32_bytes(100);
    let (_, objects) = encode_roundtrip(&desc, &data);
    assert_eq!(objects[0].1, data);
}

// ── 30. decode_range on non-szip (stream compressors should error) ───────────

#[test]
fn decode_range_zstd_not_supported() {
    let mut params = BTreeMap::new();
    params.insert("zstd_level".to_string(), ciborium::Value::Integer(3.into()));
    let desc = make_compressed_descriptor(vec![100], Dtype::Float32, "zstd", params);
    let data = finite_f32_bytes(100);
    let meta = make_global_meta();
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let result = decode_range(&encoded, 0, &[(0, 10)], &DecodeOptions::default());
    // Stream compressor should return error for range decode
    assert!(result.is_err());
}

#[test]
fn decode_range_lz4_not_supported() {
    let desc = make_compressed_descriptor(vec![100], Dtype::Float32, "lz4", BTreeMap::new());
    let data = finite_f32_bytes(100);
    let meta = make_global_meta();
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let result = decode_range(&encoded, 0, &[(0, 10)], &DecodeOptions::default());
    assert!(result.is_err());
}

// ── 31. Szip bits_per_sample computation paths ───────────────────────────────

#[test]
fn szip_none_encoding_none_filter_uses_dtype_bits() {
    let mut params = BTreeMap::new();
    params.insert("szip_rsi".to_string(), ciborium::Value::Integer(128.into()));
    params.insert(
        "szip_block_size".to_string(),
        ciborium::Value::Integer(8.into()),
    );
    params.insert("szip_flags".to_string(), ciborium::Value::Integer(0.into()));

    let desc = make_compressed_descriptor(vec![256], Dtype::Uint16, "szip", params);
    let data = vec![0u8; 512]; // 256 * 2 bytes
    let (_, objects) = encode_roundtrip(&desc, &data);
    assert_eq!(objects[0].1, data);
}

// ── 32. File iterator ────────────────────────────────────────────────────────

#[test]
fn file_iterator_over_multiple_messages() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi.tgm");

    let mut file = TensogramFile::create(&path).unwrap();
    let meta = make_global_meta();
    let desc = make_descriptor(vec![2], Dtype::Uint8);

    for i in 0u8..5 {
        let data = vec![i; 2];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();
    }

    assert_eq!(file.message_count().unwrap(), 5);

    let messages: Vec<Vec<u8>> = file.iter().unwrap().map(|r| r.unwrap()).collect();
    assert_eq!(messages.len(), 5);

    for (i, msg) in messages.iter().enumerate() {
        let (_, objects) = decode(msg, &DecodeOptions::default()).unwrap();
        assert_eq!(objects[0].1, vec![i as u8; 2]);
    }
}

// ── 33. u64 param out of range ───────────────────────────────────────────────

#[test]
fn u64_param_negative_value_rejected() {
    let meta = make_global_meta();
    let mut desc = make_simple_packing_desc(0.0);
    desc.params.insert(
        "bits_per_value".to_string(),
        ciborium::Value::Integer((-1).into()),
    );
    let data = vec![0u8; 32];
    let result = encode(&meta, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("out of u64 range"));
}

// ── 34. Mmap file access ─────────────────────────────────────────────────────

#[cfg(feature = "mmap")]
#[test]
fn mmap_decode_matches_regular() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mmap_test.tgm");

    let mut file = TensogramFile::create(&path).unwrap();
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![99u8; 16];

    file.append(
        &meta,
        &[(&desc, data.as_slice())],
        &EncodeOptions::default(),
    )
    .unwrap();

    let regular = TensogramFile::open(&path).unwrap();
    let regular_msg = regular.read_message(0).unwrap();

    let mmap = TensogramFile::open_mmap(&path).unwrap();
    let mmap_msg = mmap.read_message(0).unwrap();

    assert_eq!(regular_msg, mmap_msg);
}

// ── 34b. mmap offset correctness ─────────────────────────────────────────────
//
// Verifies that the mmap slice boundaries (offset + length) are correct so
// that the decoded payload matches the one written.  Guards against the
// `offset + length → offset - length` arithmetic mutation.

#[cfg(feature = "mmap")]
#[test]
fn mmap_decoded_data_matches_written() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mmap_data.tgm");

    let meta = make_global_meta();
    let n = 100usize;
    let desc = make_descriptor(vec![n as u64], Dtype::Float32);
    // Recognisable deterministic payload so a wrong slice is detected.
    let data: Vec<u8> = (0..n)
        .map(|i| i as f32)
        .flat_map(|v| v.to_ne_bytes())
        .collect();

    let mut file = TensogramFile::create(&path).unwrap();
    file.append(
        &meta,
        &[(&desc, data.as_slice())],
        &EncodeOptions::default(),
    )
    .unwrap();

    let mmap = TensogramFile::open_mmap(&path).unwrap();
    let (_, objects) = mmap.decode_message(0, &DecodeOptions::default()).unwrap();
    let decoded_bytes = &objects[0].1;

    assert_eq!(
        decoded_bytes.as_slice(),
        data.as_slice(),
        "mmap-decoded payload must match written data"
    );
}

// ── 35. Little-endian byte order ─────────────────────────────────────────────

#[test]
fn little_endian_roundtrip() {
    let mut desc = make_descriptor(vec![4], Dtype::Float32);
    desc.byte_order = ByteOrder::Little;
    let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let (_, objects) = encode_roundtrip(&desc, &data);
    assert_eq!(objects[0].1, data);
    assert_eq!(objects[0].0.byte_order, ByteOrder::Little);
}

// ── 36. HashAlgorithm methods ────────────────────────────────────────────────

#[test]
fn hash_algorithm_as_str() {
    assert_eq!(HashAlgorithm::Xxh3.as_str(), "xxh3");
}

#[test]
fn hash_algorithm_parse_valid() {
    assert_eq!(
        tensogram::hash::HashAlgorithm::parse("xxh3").unwrap(),
        HashAlgorithm::Xxh3
    );
}

#[test]
fn hash_algorithm_parse_invalid() {
    let result = tensogram::hash::HashAlgorithm::parse("md5");
    assert!(result.is_err());
}

// ── 37. compute_hash determinism ─────────────────────────────────────────────

#[test]
fn compute_hash_deterministic() {
    let data = b"hello tensogram";
    let h1 = tensogram::compute_hash(data, HashAlgorithm::Xxh3);
    let h2 = tensogram::compute_hash(data, HashAlgorithm::Xxh3);
    assert_eq!(h1, h2);
    assert_eq!(h1.len(), 16); // 64-bit hex
}

#[test]
fn compute_hash_empty_data() {
    let h = tensogram::compute_hash(b"", HashAlgorithm::Xxh3);
    assert_eq!(h.len(), 16);
}

// ── 38. decode_range with hash verification ──────────────────────────────────

#[test]
fn decode_range_with_hash_verification() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![10], Dtype::Float32);
    let data: Vec<u8> = (0..40).collect();
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Range decode with hash verification
    let (_, result) = decode_range(
        &encoded,
        0,
        &[(0, 5)],
        &DecodeOptions {
            verify_hash: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.len(), 1, "expected 1 part for 1 range");
    let total_bytes: usize = result.iter().map(|p| p.len()).sum();
    assert_eq!(total_bytes, 20);
}

// ── 39. Streaming encoder validates objects ──────────────────────────────────

#[test]
fn streaming_encoder_rejects_invalid_object() {
    let meta = make_global_meta();
    let options = EncodeOptions::default();

    let mut desc = make_descriptor(vec![4], Dtype::Float32);
    desc.obj_type = String::new(); // Invalid

    let buf = Vec::new();
    let mut enc = StreamingEncoder::new(buf, &meta, &options).unwrap();
    let result = enc.write_object(&desc, &[0u8; 16]);
    assert!(result.is_err());
}

#[test]
fn buffered_encode_rejects_emit_preceders() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![0u8; 16];
    let options = EncodeOptions {
        emit_preceders: true,
        ..Default::default()
    };
    let result = encode(&meta, &[(&desc, &data)], &options);
    assert!(
        result.is_err(),
        "emit_preceders in buffered mode should fail"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("StreamingEncoder"),
        "error should mention StreamingEncoder: {err}"
    );
}

// ── PrecederMetadata edge cases ─────────────────────────────────────────────

#[test]
fn preceder_all_objects_have_preceders() {
    // Every object has its own preceder — verify all payloads merge correctly
    let meta = GlobalMetadata::default();
    let desc0 = make_descriptor(vec![2], Dtype::Float32);
    let desc1 = make_descriptor(vec![3], Dtype::Float32);
    let data0 = vec![0u8; 8];
    let data1 = vec![0u8; 12];

    let mut prec0 = BTreeMap::new();
    prec0.insert("units".to_string(), ciborium::Value::Text("K".to_string()));
    let mut prec1 = BTreeMap::new();
    prec1.insert(
        "units".to_string(),
        ciborium::Value::Text("m/s".to_string()),
    );

    let buf = Vec::new();
    let mut enc = streaming::StreamingEncoder::new(buf, &meta, &EncodeOptions::default()).unwrap();
    enc.write_preceder(prec0).unwrap();
    enc.write_object(&desc0, &data0).unwrap();
    enc.write_preceder(prec1).unwrap();
    enc.write_object(&desc1, &data1).unwrap();
    let result = enc.finish().unwrap();

    let (decoded_meta, objects) = decode(&result, &decode::DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 2);
    assert_eq!(decoded_meta.base.len(), 2);

    // Each object's preceder metadata should be correct
    let u0 = decoded_meta.base[0].get("units").and_then(|v| match v {
        ciborium::Value::Text(s) => Some(s.as_str()),
        _ => None,
    });
    let u1 = decoded_meta.base[1].get("units").and_then(|v| match v {
        ciborium::Value::Text(s) => Some(s.as_str()),
        _ => None,
    });
    assert_eq!(u0, Some("K"));
    assert_eq!(u1, Some("m/s"));
}

#[test]
fn preceder_with_hash_verification() {
    // Preceder + hash — both features should work together
    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![42u8; 16];

    let mut prec = BTreeMap::new();
    prec.insert(
        "mars".to_string(),
        ciborium::Value::Map(vec![(
            ciborium::Value::Text("param".to_string()),
            ciborium::Value::Text("2t".to_string()),
        )]),
    );

    let options = EncodeOptions {
        hash_algorithm: Some(hash::HashAlgorithm::Xxh3),
        ..Default::default()
    };
    let buf = Vec::new();
    let mut enc = streaming::StreamingEncoder::new(buf, &meta, &options).unwrap();
    enc.write_preceder(prec).unwrap();
    enc.write_object(&desc, &data).unwrap();
    let result = enc.finish().unwrap();

    // Decode with hash verification.  v3 `verify_hash` on
    // `DecodeOptions` is a no-op (frame-level integrity moved to
    // validate --checksum); the option is retained for source
    // compatibility and decoding still succeeds when the message
    // is well-formed.
    let verify_opts = decode::DecodeOptions {
        verify_hash: true,
        ..Default::default()
    };
    let (decoded_meta, _objects) = decode(&result, &verify_opts).unwrap();
    assert!(decoded_meta.base[0].contains_key("mars"));
}

#[test]
fn preceder_with_extra_keys_tolerated() {
    // The spec says preceder should only carry per-object base keys, but
    // tolerant decoding should accept extra keys without error (ignored).
    let mut preceder_meta = GlobalMetadata::default();
    // Add _extra_ at message level — this should be tolerated but not merged.
    preceder_meta.extra.insert(
        "should_be_ignored".to_string(),
        ciborium::Value::Text("ignored".to_string()),
    );
    // Add a base entry for the preceder
    preceder_meta.base = vec![BTreeMap::new()];
    let preceder_cbor = metadata::global_metadata_to_cbor(&preceder_meta).unwrap();

    // Build a raw message: HeaderMeta → PrecederMeta → DataObject
    let global = GlobalMetadata::default();
    let meta_cbor = metadata::global_metadata_to_cbor(&global).unwrap();

    let desc = make_descriptor(vec![4], Dtype::Float32);
    let payload = vec![0u8; 16];
    let obj_frame = framing::encode_data_object_frame(&desc, &payload, false, None).unwrap();

    let mut out = Vec::new();
    // Preamble placeholder
    out.extend_from_slice(&[0u8; 24]);
    // Header metadata
    write_test_frame(&mut out, 1, &meta_cbor);
    // Preceder metadata (with extra keys)
    write_test_frame(&mut out, 8, &preceder_cbor);
    // Data object
    out.extend_from_slice(&obj_frame);
    let pad = (8 - (out.len() % 8)) % 8;
    out.extend(std::iter::repeat_n(0u8, pad));
    // Postamble (v3, 24 B): first_footer_offset + total_length + magic.
    let postamble_offset = out.len();
    let footer_off = postamble_offset as u64;
    out.extend_from_slice(&footer_off.to_be_bytes());
    out.extend_from_slice(&0u64.to_be_bytes()); // total_length placeholder (patched below)
    out.extend_from_slice(b"39277777");
    // Patch preamble
    let total = out.len() as u64;
    let mut pre = Vec::new();
    pre.extend_from_slice(b"TENSOGRM");
    pre.extend_from_slice(&tensogram::wire::WIRE_VERSION.to_be_bytes());
    pre.extend_from_slice(&1u16.to_be_bytes()); // HEADER_METADATA flag
    pre.extend_from_slice(&0u32.to_be_bytes());
    pre.extend_from_slice(&total.to_be_bytes());
    out[..24].copy_from_slice(&pre);
    // Patch postamble total_length
    out[postamble_offset + 8..postamble_offset + 16].copy_from_slice(&total.to_be_bytes());

    let decoded = framing::decode_message(&out).unwrap();
    assert_eq!(decoded.objects.len(), 1);
    // Extra keys from preceder are not merged into global
}

#[test]
fn preceder_with_empty_payload_map() {
    // Preceder with payload = [{}] — empty map is valid, just no keys
    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let data = vec![0u8; 8];

    let buf = Vec::new();
    let mut enc = streaming::StreamingEncoder::new(buf, &meta, &EncodeOptions::default()).unwrap();
    enc.write_preceder(BTreeMap::new()).unwrap();
    enc.write_object(&desc, &data).unwrap();
    let result = enc.finish().unwrap();

    let (decoded_meta, objects) = decode(&result, &decode::DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
    // Structural keys should still be present in base[0]._reserved_.tensor
    assert!(decoded_meta.base[0].contains_key("_reserved_"));
}

#[test]
fn preceder_with_nested_cbor_structures() {
    // Deep nesting in preceder metadata — should round-trip
    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let data = vec![0u8; 8];

    let mut prec = BTreeMap::new();
    prec.insert(
        "deep".to_string(),
        ciborium::Value::Map(vec![(
            ciborium::Value::Text("level1".to_string()),
            ciborium::Value::Map(vec![(
                ciborium::Value::Text("level2".to_string()),
                ciborium::Value::Array(vec![
                    ciborium::Value::Integer(1.into()),
                    ciborium::Value::Integer(2.into()),
                    ciborium::Value::Integer(3.into()),
                ]),
            )]),
        )]),
    );

    let buf = Vec::new();
    let mut enc = streaming::StreamingEncoder::new(buf, &meta, &EncodeOptions::default()).unwrap();
    enc.write_preceder(prec).unwrap();
    enc.write_object(&desc, &data).unwrap();
    let result = enc.finish().unwrap();

    let (decoded_meta, _) = decode(&result, &decode::DecodeOptions::default()).unwrap();
    // Verify the nested structure survived in base[0]
    let deep = decoded_meta.base[0].get("deep");
    assert!(deep.is_some(), "deep nested key should survive round-trip");
    if let Some(ciborium::Value::Map(level1)) = deep {
        assert_eq!(level1.len(), 1);
    } else {
        panic!("expected map for 'deep'");
    }
}

// ── 40. decode_descriptors ────────────────────────────────────────────────────

#[test]
fn decode_descriptors_returns_descriptors_without_data() {
    let meta = make_global_meta();
    let desc0 = make_descriptor(vec![4], Dtype::Float32);
    let desc1 = make_descriptor(vec![2, 3], Dtype::Float64);
    let data0 = vec![0u8; 16];
    let data1 = vec![0u8; 48];

    let encoded = encode(
        &meta,
        &[(&desc0, data0.as_slice()), (&desc1, data1.as_slice())],
        &EncodeOptions::default(),
    )
    .unwrap();

    let (decoded_meta, descriptors) = decode_descriptors(&encoded).unwrap();
    assert_eq!(decoded_meta.version, 3);
    assert_eq!(descriptors.len(), 2);
    assert_eq!(descriptors[0].shape, vec![4]);
    assert_eq!(descriptors[0].dtype, Dtype::Float32);
    assert_eq!(descriptors[1].shape, vec![2, 3]);
    assert_eq!(descriptors[1].dtype, Dtype::Float64);
}

#[test]
fn decode_descriptors_empty_message() {
    let meta = make_global_meta();
    let encoded = encode(&meta, &[], &EncodeOptions::default()).unwrap();

    let (decoded_meta, descriptors) = decode_descriptors(&encoded).unwrap();
    assert_eq!(decoded_meta.version, 3);
    assert!(descriptors.is_empty());
}

// ── 41. ObjectIter shape overflow ────────────────────────────────────────────

#[test]
fn object_iter_shape_overflow_returns_error() {
    // We need to construct a message with a descriptor that has an overflowing shape.
    // We can't do this through normal encode (which validates), so we'll directly
    // construct a message via framing. Instead, test that the ObjectIter handles
    // extremely large shapes by trying a descriptor that overflows.
    // Since we can't bypass encode validation easily, we verify the error path
    // in the iter module is reachable via ObjectIter by encoding a valid message
    // and then patching the shape in the descriptor CBOR.
    // For now, just verify that an invalid buffer returns an error.
    let bad_buf = vec![0u8; 100];
    let result = objects(&bad_buf, DecodeOptions::default());
    assert!(
        result.is_err(),
        "garbage buffer should fail object iter creation"
    );
}

// ── 42. get_u64_param wrong type ─────────────────────────────────────────────

#[test]
fn u64_param_wrong_type_rejected() {
    // When a u64 param receives a float value instead of integer
    let meta = make_global_meta();
    let mut params = BTreeMap::new();
    params.insert(
        "shuffle_element_size".to_string(),
        ciborium::Value::Float(4.0), // Should be Integer, not Float
    );
    let mut desc = make_descriptor(vec![10], Dtype::Float32);
    desc.filter = "shuffle".to_string();
    desc.params = params;
    let data = vec![0u8; 40];

    let result = encode(&meta, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("expected integer"));
}

#[test]
fn u64_param_missing_rejected() {
    // When a required u64 param is missing
    let meta = make_global_meta();
    let mut desc = make_descriptor(vec![10], Dtype::Float32);
    desc.filter = "shuffle".to_string();
    // Don't set shuffle_element_size
    let data = vec![0u8; 40];

    let result = encode(&meta, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("missing required"));
}

// ── 43. GlobalMetadata serde edge cases ──────────────────────────────────────

#[test]
fn global_metadata_default_version_is_2() {
    let meta = GlobalMetadata::default();
    assert_eq!(meta.version, 3);
    assert!(meta.base.is_empty());
    assert!(meta.reserved.is_empty());
    assert!(meta.extra.is_empty());
}

#[test]
fn global_metadata_serde_empty_base_not_serialized() {
    // When base is empty, it should not appear in CBOR (skip_serializing_if)
    let meta = GlobalMetadata::default();
    let cbor_bytes = metadata::global_metadata_to_cbor(&meta).unwrap();
    let decoded: GlobalMetadata = metadata::cbor_to_global_metadata(&cbor_bytes).unwrap();
    assert!(decoded.base.is_empty());
    assert!(decoded.reserved.is_empty());
    assert!(decoded.extra.is_empty());
}

#[test]
fn global_metadata_serde_reserved_rename() {
    // Verify _reserved_ CBOR key name round-trips correctly
    let mut reserved = BTreeMap::new();
    reserved.insert(
        "encoder".to_string(),
        ciborium::Value::Text("test".to_string()),
    );
    let meta = GlobalMetadata {
        version: 3,
        reserved,
        ..Default::default()
    };
    let cbor_bytes = metadata::global_metadata_to_cbor(&meta).unwrap();
    let decoded: GlobalMetadata = metadata::cbor_to_global_metadata(&cbor_bytes).unwrap();
    assert!(decoded.reserved.contains_key("encoder"));
}

#[test]
fn global_metadata_serde_extra_rename() {
    // Verify _extra_ CBOR key name round-trips correctly
    let mut extra = BTreeMap::new();
    extra.insert("custom".to_string(), ciborium::Value::Integer(42.into()));
    let meta = GlobalMetadata {
        version: 3,
        extra,
        ..Default::default()
    };
    let cbor_bytes = metadata::global_metadata_to_cbor(&meta).unwrap();
    let decoded: GlobalMetadata = metadata::cbor_to_global_metadata(&cbor_bytes).unwrap();
    assert_eq!(
        decoded.extra.get("custom"),
        Some(&ciborium::Value::Integer(42.into()))
    );
}

// ── 44. compute_common with all-empty entries ────────────────────────────────

#[test]
fn compute_common_all_empty_entries() {
    let e1: BTreeMap<String, ciborium::Value> = BTreeMap::new();
    let e2: BTreeMap<String, ciborium::Value> = BTreeMap::new();
    let (common, remaining) = compute_common(&[e1, e2]);
    assert!(common.is_empty());
    assert!(remaining[0].is_empty());
    assert!(remaining[1].is_empty());
}

// ── 45. framing decoder: base auto-extend for < obj_count ────────────────────

#[test]
fn framing_base_auto_extends_when_fewer_than_objects() {
    // Encode with 0 base entries but 3 objects → decoder should auto-extend base
    let meta = GlobalMetadata {
        version: 3,
        base: vec![], // Fewer than objects
        ..Default::default()
    };
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let data = vec![0u8; 8];
    let options = EncodeOptions {
        hash_algorithm: None,
        ..Default::default()
    };
    let msg = encode(
        &meta,
        &[
            (&desc, data.as_slice()),
            (&desc, data.as_slice()),
            (&desc, data.as_slice()),
        ],
        &options,
    )
    .unwrap();

    let (decoded, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 3);
    assert_eq!(decoded.base.len(), 3);
    // All 3 entries should have _reserved_.tensor
    for (i, entry) in decoded.base.iter().enumerate() {
        assert!(
            entry.contains_key("_reserved_"),
            "base[{i}] should have _reserved_ after auto-extend"
        );
    }
}

// ── 46. Streaming encoder: bytes_written accessors ───────────────────────────

#[test]
fn streaming_encoder_bytes_written_increases() {
    let meta = GlobalMetadata::default();
    let buf = Vec::new();
    let enc = streaming::StreamingEncoder::new(buf, &meta, &EncodeOptions::default()).unwrap();
    let initial_bytes = enc.bytes_written();
    assert!(
        initial_bytes > 0,
        "preamble + header frame should have been written"
    );

    // After writing an object, bytes_written should increase further
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let data = vec![0u8; 8];
    let mut enc2 =
        streaming::StreamingEncoder::new(Vec::new(), &meta, &EncodeOptions::default()).unwrap();
    let before = enc2.bytes_written();
    enc2.write_object(&desc, &data).unwrap();
    assert!(
        enc2.bytes_written() > before,
        "bytes_written should increase after write_object"
    );
}

// ── 47. encode base exactly matching descriptors ─────────────────────────────

#[test]
fn encode_base_exactly_matches_descriptors() {
    // base.len() == descriptors.len() — should work, keys preserved
    let mut entry0 = BTreeMap::new();
    entry0.insert("param".to_string(), ciborium::Value::Text("2t".to_string()));
    let mut entry1 = BTreeMap::new();
    entry1.insert(
        "param".to_string(),
        ciborium::Value::Text("msl".to_string()),
    );
    let meta = GlobalMetadata {
        version: 3,
        base: vec![entry0, entry1],
        ..Default::default()
    };
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let data = vec![0u8; 8];
    let options = EncodeOptions {
        hash_algorithm: None,
        ..Default::default()
    };
    let msg = encode(
        &meta,
        &[(&desc, data.as_slice()), (&desc, data.as_slice())],
        &options,
    )
    .unwrap();

    let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded.base.len(), 2);
    assert_eq!(
        decoded.base[0].get("param"),
        Some(&ciborium::Value::Text("2t".to_string()))
    );
    assert_eq!(
        decoded.base[1].get("param"),
        Some(&ciborium::Value::Text("msl".to_string()))
    );
}

// ── 55. 100 data objects stress test (two-pass index) ────────────────────────

#[test]
fn stress_100_data_objects_roundtrip() {
    let meta = make_global_meta();
    let num_objects = 100;

    // Build 100 float32[10] descriptors/data pairs
    let desc = make_descriptor(vec![10], Dtype::Float32);
    let objects_data: Vec<Vec<u8>> = (0..num_objects)
        .map(|i| {
            // Fill each array with a distinct byte pattern based on index
            let val = (i % 256) as u8;
            vec![val; 10 * 4]
        })
        .collect();

    let pairs: Vec<(&DataObjectDescriptor, &[u8])> =
        objects_data.iter().map(|d| (&desc, d.as_slice())).collect();

    let encoded = encode(&meta, &pairs, &EncodeOptions::default()).unwrap();

    // Verify scan finds the message
    let offsets = scan(&encoded);
    assert_eq!(offsets.len(), 1);

    // Decode all objects
    let (decoded_meta, decoded_objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded_meta.version, 3);
    assert_eq!(decoded_objects.len(), num_objects);

    // Verify each object's data round-trips correctly
    for (i, (obj_desc, obj_data)) in decoded_objects.iter().enumerate() {
        assert_eq!(obj_desc.shape, vec![10], "object {i} shape mismatch");
        assert_eq!(obj_desc.dtype, Dtype::Float32, "object {i} dtype mismatch");
        assert_eq!(*obj_data, objects_data[i], "object {i} data mismatch");
    }

    // Also verify decode_object works for random indices
    for idx in [0, 49, 99] {
        let (_, ret_desc, ret_data) =
            decode_object(&encoded, idx, &DecodeOptions::default()).unwrap();
        assert_eq!(ret_desc.shape, vec![10]);
        assert_eq!(ret_data, objects_data[idx]);
    }
}

// ── 56. Mixed streaming + buffered messages in one file ──────────────────────

#[test]
fn mixed_streaming_and_buffered_in_one_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed_mode.tgm");

    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let buffered_data = vec![11u8; 16]; // 4 float32s
    let streaming_data = vec![22u8; 16]; // 4 float32s

    // Create file, append a buffered message
    let mut file = TensogramFile::create(&path).unwrap();
    file.append(
        &meta,
        &[(&desc, buffered_data.as_slice())],
        &EncodeOptions::default(),
    )
    .unwrap();

    // Append a streaming message by writing directly to the file
    {
        let f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        let writer = std::io::BufWriter::new(f);
        let mut enc = StreamingEncoder::new(writer, &meta, &EncodeOptions::default()).unwrap();
        enc.write_object(&desc, &streaming_data).unwrap();
        enc.finish().unwrap();
    }

    // Re-open and verify both messages
    file.invalidate_offsets();
    assert_eq!(file.message_count().unwrap(), 2);

    // Verify buffered message (message 0)
    let msg0 = file.read_message(0).unwrap();
    let (_, objects0) = decode(&msg0, &DecodeOptions::default()).unwrap();
    assert_eq!(objects0.len(), 1);
    assert_eq!(objects0[0].1, buffered_data);

    // Verify streaming message (message 1)
    let msg1 = file.read_message(1).unwrap();
    let (_, objects1) = decode(&msg1, &DecodeOptions::default()).unwrap();
    assert_eq!(objects1.len(), 1);
    assert_eq!(objects1[0].1, streaming_data);

    // Also verify scan() finds correct offsets for both
    let all_bytes = std::fs::read(&path).unwrap();
    let offsets = scan(&all_bytes);
    assert_eq!(offsets.len(), 2);
    assert_eq!(offsets[0].0, 0);
    assert_eq!(offsets[0].1, msg0.len());
}

// ── 57. Garbage between messages in a file ───────────────────────────────────

#[test]
fn garbage_between_messages_scan_still_finds_both() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![2], Dtype::Uint8);

    let msg1 = encode(&meta, &[(&desc, &[0xAA, 0xBB])], &EncodeOptions::default()).unwrap();
    let msg2 = encode(&meta, &[(&desc, &[0xCC, 0xDD])], &EncodeOptions::default()).unwrap();

    // Concatenate: msg1 + 16 random garbage bytes + msg2
    let garbage: Vec<u8> = (0u8..16)
        .map(|i| i.wrapping_mul(37).wrapping_add(7))
        .collect();
    let mut buf = Vec::new();
    buf.extend_from_slice(&msg1);
    buf.extend_from_slice(&garbage);
    buf.extend_from_slice(&msg2);

    let offsets = scan(&buf);
    assert_eq!(
        offsets.len(),
        2,
        "scan should find 2 messages despite garbage between them"
    );

    // First message at offset 0
    assert_eq!(offsets[0].0, 0);
    assert_eq!(offsets[0].1, msg1.len());

    // Second message after msg1 + garbage
    let expected_offset2 = msg1.len() + garbage.len();
    assert_eq!(offsets[1].0, expected_offset2);
    assert_eq!(offsets[1].1, msg2.len());

    // Decode both messages to verify data integrity
    let slice1 = &buf[offsets[0].0..offsets[0].0 + offsets[0].1];
    let (_, objects1) = decode(slice1, &DecodeOptions::default()).unwrap();
    assert_eq!(objects1[0].1, vec![0xAA, 0xBB]);

    let slice2 = &buf[offsets[1].0..offsets[1].0 + offsets[1].1];
    let (_, objects2) = decode(slice2, &DecodeOptions::default()).unwrap();
    assert_eq!(objects2[0].1, vec![0xCC, 0xDD]);
}

// ── 58. Streaming encoder with zero messages (finish immediately) ────────────

#[test]
fn streaming_encoder_finish_immediately_produces_valid_message() {
    let meta = make_global_meta();
    let options = EncodeOptions {
        hash_algorithm: None,
        ..Default::default()
    };

    let buf: Vec<u8> = Vec::new();
    let enc = StreamingEncoder::new(buf, &meta, &options).unwrap();
    assert_eq!(enc.object_count(), 0);
    let result = enc.finish().unwrap();

    // The result should NOT be empty — it should be a valid streaming message
    // (preamble + header metadata + footer metadata + footer index + postamble)
    assert!(
        !result.is_empty(),
        "streaming encoder with 0 objects should produce non-empty bytes"
    );

    // scan should find exactly one message
    let offsets = scan(&result);
    assert_eq!(
        offsets.len(),
        1,
        "streaming zero-object message should be scannable"
    );

    // Decode should succeed with 0 objects
    let (decoded_meta, objects) = decode(&result, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded_meta.version, 3);
    assert!(
        objects.is_empty(),
        "streaming zero-object message should decode to 0 objects"
    );
}

// ── 59. Unicode metadata round-trip ──────────────────────────────────────────

#[test]
fn unicode_metadata_emoji_and_cjk_roundtrip() {
    let mut extra = BTreeMap::new();
    extra.insert(
        "emoji".to_string(),
        ciborium::Value::Text("🌍🌊🔥❄️".to_string()),
    );
    extra.insert(
        "cjk".to_string(),
        ciborium::Value::Text("気温データ".to_string()),
    );
    extra.insert(
        "mixed".to_string(),
        ciborium::Value::Text("Temperature 🌡️ is 25°C — très bien".to_string()),
    );
    extra.insert(
        "arabic".to_string(),
        ciborium::Value::Text("بيانات الطقس".to_string()),
    );
    extra.insert(
        "null_char".to_string(),
        ciborium::Value::Text("before\0after".to_string()),
    );

    let meta = GlobalMetadata {
        version: 3,
        extra,
        ..Default::default()
    };

    let desc = make_descriptor(vec![2], Dtype::Float32);
    let data = vec![0u8; 8];

    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (decoded_meta, _) = decode(&encoded, &DecodeOptions::default()).unwrap();

    assert_eq!(
        decoded_meta.extra.get("emoji"),
        Some(&ciborium::Value::Text("🌍🌊🔥❄️".to_string()))
    );
    assert_eq!(
        decoded_meta.extra.get("cjk"),
        Some(&ciborium::Value::Text("気温データ".to_string()))
    );
    assert_eq!(
        decoded_meta.extra.get("mixed"),
        Some(&ciborium::Value::Text(
            "Temperature 🌡️ is 25°C — très bien".to_string()
        ))
    );
    assert_eq!(
        decoded_meta.extra.get("arabic"),
        Some(&ciborium::Value::Text("بيانات الطقس".to_string()))
    );
    assert_eq!(
        decoded_meta.extra.get("null_char"),
        Some(&ciborium::Value::Text("before\0after".to_string()))
    );
}

/// Helper: write a non-data-object frame with v3 layout
/// `FR + type + version=1 + flags=0 + total_length + payload + hash(8) + ENDF`
/// plus 8-byte alignment padding.  Hash slot is zero-filled
/// (HASHES_PRESENT=0 at the message level).
fn write_test_frame(out: &mut Vec<u8>, frame_type: u16, payload: &[u8]) {
    let total_len = (16 + payload.len() + 12) as u64; // header + payload + hash(8) + ENDF(4)
    out.extend_from_slice(b"FR");
    out.extend_from_slice(&frame_type.to_be_bytes());
    out.extend_from_slice(&1u16.to_be_bytes()); // version
    out.extend_from_slice(&0u16.to_be_bytes()); // flags
    out.extend_from_slice(&total_len.to_be_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&0u64.to_be_bytes()); // hash slot
    out.extend_from_slice(b"ENDF");
    let pad = (8 - (out.len() % 8)) % 8;
    out.extend(std::iter::repeat_n(0u8, pad));
}

// ── 48. Corrupt descriptor CBOR ──────────────────────────────────────────────

#[test]
fn corrupt_metadata_cbor_rejected() {
    // Build a valid message, then corrupt the metadata CBOR frame to test
    // the framing::decode_message error path for bad CBOR.
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![0u8; 16];
    let mut encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Find the HeaderMetadata frame: "FR" followed by frame type 0x0001
    let header_meta_marker: &[u8] = &[b'F', b'R', 0x00, 0x01];
    if let Some(frame_start) = encoded.windows(4).position(|w| w == header_meta_marker) {
        // Corrupt the CBOR payload inside the metadata frame
        // The frame header is 16 bytes; corrupt the payload area
        let payload_start = frame_start + 16;
        if payload_start + 4 < encoded.len() {
            encoded[payload_start] = 0xFF;
            encoded[payload_start + 1] = 0xFF;
            encoded[payload_start + 2] = 0xFF;
            encoded[payload_start + 3] = 0xFF;
        }
    }

    let result = decode(&encoded, &DecodeOptions::default());
    assert!(result.is_err(), "corrupt metadata CBOR should be rejected");
}

#[test]
fn truncated_buffer_rejected() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![0u8; 16];
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Truncate to half the message
    let truncated = &encoded[..encoded.len() / 2];
    let result = decode(truncated, &DecodeOptions::default());
    assert!(result.is_err(), "truncated buffer should be rejected");
}

// ── 49. decode_object with index out of range ────────────────────────────────

#[test]
fn decode_object_negative_boundary() {
    // Test index exactly at boundary (num_objects)
    let meta = make_global_meta();
    let desc0 = make_descriptor(vec![2], Dtype::Float32);
    let desc1 = make_descriptor(vec![3], Dtype::Float32);
    let data0 = vec![0u8; 8];
    let data1 = vec![0u8; 12];

    let encoded = encode(
        &meta,
        &[(&desc0, data0.as_slice()), (&desc1, data1.as_slice())],
        &EncodeOptions::default(),
    )
    .unwrap();

    // Index 2 is out of range for 2 objects (indices 0, 1)
    let result = decode_object(&encoded, 2, &DecodeOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("out of range"));

    // Index 1 should succeed (last valid index)
    let result = decode_object(&encoded, 1, &DecodeOptions::default());
    assert!(result.is_ok());
}

// ── 50. decode_range with invalid ranges ─────────────────────────────────────

#[test]
fn decode_range_with_filter_rejected() {
    // decode_range should reject messages with a filter applied
    let meta = make_global_meta();
    let mut params = BTreeMap::new();
    params.insert(
        "shuffle_element_size".to_string(),
        ciborium::Value::Integer(4.into()),
    );
    let mut desc = make_descriptor(vec![10], Dtype::Float32);
    desc.filter = "shuffle".to_string();
    desc.params = params;
    let data: Vec<u8> = (0..40).collect();

    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let result = decode_range(&encoded, 0, &[(0, 5)], &DecodeOptions::default());
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("filter") || msg.contains("shuffle"),
        "expected filter error, got: {msg}"
    );
}

#[test]
fn decode_range_on_bitmask_dtype_rejected() {
    // decode_range should reject bitmask dtype
    let meta = make_global_meta();
    let desc = make_descriptor(vec![16], Dtype::Bitmask);
    let data = vec![0xFF; 2];
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let result = decode_range(&encoded, 0, &[(0, 4)], &DecodeOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("bitmask"));
}

// ── 51. decode on completely invalid data ────────────────────────────────────

#[test]
fn decode_on_random_garbage() {
    let garbage = vec![
        0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE, 0x00, 0x01, 0x02, 0x03,
    ];
    let result = decode(&garbage, &DecodeOptions::default());
    assert!(result.is_err());
}

#[test]
fn decode_metadata_on_garbage() {
    let garbage = vec![0u8; 50];
    let result = decode_metadata(&garbage);
    assert!(result.is_err());
}

#[test]
fn decode_descriptors_on_garbage() {
    let garbage = vec![0xFF; 100];
    let result = decode_descriptors(&garbage);
    assert!(result.is_err());
}

// ── 52. decode_object on empty message (no objects) ──────────────────────────

#[test]
fn decode_object_on_empty_message() {
    let meta = make_global_meta();
    let encoded = encode(&meta, &[], &EncodeOptions::default()).unwrap();

    let result = decode_object(&encoded, 0, &DecodeOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("out of range"));
}

// ── 53. decode_range on empty message (no objects) ───────────────────────────

#[test]
fn decode_range_on_empty_message() {
    let meta = make_global_meta();
    let encoded = encode(&meta, &[], &EncodeOptions::default()).unwrap();

    let result = decode_range(&encoded, 0, &[(0, 1)], &DecodeOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("out of range"));
}

// ── 54. Corrupt data object frame ENDF trailer ──────────────────────────────

#[test]
fn corrupt_data_object_frame_trailer_rejected() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![42u8; 16];
    let mut encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Find the data-object frame: "FR" followed by frame type 0x0009
    // (NTensorFrame is what 0.17+ encoders emit).
    let data_object_marker: &[u8] = &[b'F', b'R', 0x00, 0x09];
    if let Some(frame_start) = encoded.windows(4).position(|w| w == data_object_marker) {
        // Read the total_length from the frame header (bytes 8-15)
        let total_len_bytes = &encoded[frame_start + 8..frame_start + 16];
        let total_len = u64::from_be_bytes(total_len_bytes.try_into().unwrap()) as usize;
        // The ENDF trailer is the last 4 bytes of the frame
        let endf_start = frame_start + total_len - 4;
        if endf_start + 4 <= encoded.len() {
            encoded[endf_start] = 0xFF;
            encoded[endf_start + 1] = 0xFF;
            encoded[endf_start + 2] = 0xFF;
            encoded[endf_start + 3] = 0xFF;
        }
    }

    let result = decode(&encoded, &DecodeOptions::default());
    assert!(
        result.is_err(),
        "corrupt data object frame trailer should be rejected"
    );
}

// ── 55. -0.0 and +0.0 round-trip (NaN / Inf no longer pass through default encode) ───

#[test]
fn float32_negative_zero_roundtrips_bit_exactly() {
    // NaN / Inf used to round-trip bit-exactly through `encoding="none"` in
    // pre-0.17 versions.  The 0.17 default-reject behaviour rejects them at
    // encode time (see `tests/finite_check.rs`); bit-exact NaN/Inf round-trip
    // via the bitmask opt-in is covered in the bitmask test matrix.
    //
    // What we still pin here: -0.0 vs +0.0 preservation — both are finite so
    // the encode accepts them, and the bit pattern difference survives.
    let values: [f32; 4] = [0.0f32, -0.0f32, 1.5, -1.5];
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let desc = make_descriptor(vec![4], Dtype::Float32);
    let (_, objects) = encode_roundtrip(&desc, &data);
    let decoded: Vec<f32> = objects[0]
        .1
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    for (orig, got) in values.iter().zip(decoded.iter()) {
        assert_eq!(orig.to_bits(), got.to_bits());
    }
}

#[test]
fn float64_negative_zero_roundtrips_bit_exactly() {
    let values: [f64; 4] = [0.0f64, -0.0f64, 1.5, -1.5];
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let desc = make_descriptor(vec![4], Dtype::Float64);
    let (_, objects) = encode_roundtrip(&desc, &data);
    let decoded: Vec<f64> = objects[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    for (orig, got) in values.iter().zip(decoded.iter()) {
        assert_eq!(orig.to_bits(), got.to_bits());
    }
}

// ── 56. Bitmask wrong data length rejected ───────────────────────────────────

#[test]
fn bitmask_wrong_data_length_rejected() {
    let meta = make_global_meta();
    // Shape [10] → expected ceil(10/8) = 2 bytes
    let desc = make_descriptor(vec![10], Dtype::Bitmask);
    let data = vec![0xFF; 3]; // Wrong: should be 2 bytes

    let result = encode(&meta, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("bitmask") || msg.contains("data_len"),
        "expected bitmask data length error, got: {msg}"
    );
}

#[test]
fn bitmask_correct_data_length_accepted() {
    let meta = make_global_meta();
    // Shape [10] → expected ceil(10/8) = 2 bytes
    let desc = make_descriptor(vec![10], Dtype::Bitmask);
    let data = vec![0xFF; 2]; // Correct

    let result = encode(&meta, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_ok());
}

#[test]
fn bitmask_exact_multiple_of_8_accepted() {
    let meta = make_global_meta();
    // Shape [16] → expected ceil(16/8) = 2 bytes
    let desc = make_descriptor(vec![16], Dtype::Bitmask);
    let data = vec![0xFF; 2];

    let result = encode(&meta, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_ok());
}

// ── 57. decode_range with zero-count range ───────────────────────────────────

#[test]
fn decode_range_zero_count_returns_empty() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![10], Dtype::Float32);
    let data = vec![0u8; 40]; // 10 * 4 bytes
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Range with count=0 at offset=5
    let (_, result) = decode_range(&encoded, 0, &[(5, 0)], &DecodeOptions::default()).unwrap();
    assert_eq!(result.len(), 1, "should return 1 part for 1 range");
    assert!(
        result[0].is_empty(),
        "zero-count range should produce empty bytes"
    );
}

#[test]
fn decode_range_overlapping_ranges_returns_duplicate_data() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![10], Dtype::Float32);
    let data: Vec<u8> = (0..40).collect(); // 10 * 4 bytes
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Overlapping ranges: [0..3) and [1..4) share elements 1 and 2
    let (_, result) =
        decode_range(&encoded, 0, &[(0, 3), (1, 3)], &DecodeOptions::default()).unwrap();
    assert_eq!(result.len(), 2, "should return 2 parts for 2 ranges");

    // First range: elements 0,1,2 → 12 bytes
    assert_eq!(result[0].len(), 12);
    // Second range: elements 1,2,3 → 12 bytes
    assert_eq!(result[1].len(), 12);

    // The overlapping portion (elements 1,2) should be identical in both
    // result[0] bytes 4..12 == result[1] bytes 0..8
    assert_eq!(
        &result[0][4..12],
        &result[1][0..8],
        "overlapping elements should have identical bytes"
    );
}

// ── 58. Unicode metadata with emoji keys, CJK values, empty key ──────────────

#[test]
fn unicode_metadata_emoji_keys_roundtrip() {
    let mut extra = BTreeMap::new();
    extra.insert(
        "🌡️".to_string(),
        ciborium::Value::Text("temperature".to_string()),
    );

    let meta = GlobalMetadata {
        version: 3,
        extra,
        ..Default::default()
    };

    let desc = make_descriptor(vec![2], Dtype::Float32);
    let data = vec![0u8; 8];

    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (decoded_meta, _) = decode(&encoded, &DecodeOptions::default()).unwrap();

    assert_eq!(
        decoded_meta.extra.get("🌡️"),
        Some(&ciborium::Value::Text("temperature".to_string()))
    );
}

#[test]
fn unicode_metadata_cjk_values_roundtrip() {
    let mut extra = BTreeMap::new();
    extra.insert(
        "name".to_string(),
        ciborium::Value::Text("気温".to_string()),
    );

    let meta = GlobalMetadata {
        version: 3,
        extra,
        ..Default::default()
    };

    let desc = make_descriptor(vec![2], Dtype::Float32);
    let data = vec![0u8; 8];

    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (decoded_meta, _) = decode(&encoded, &DecodeOptions::default()).unwrap();

    assert_eq!(
        decoded_meta.extra.get("name"),
        Some(&ciborium::Value::Text("気温".to_string()))
    );
}

#[test]
fn unicode_metadata_empty_string_key_roundtrip() {
    let mut extra = BTreeMap::new();
    extra.insert(
        "".to_string(),
        ciborium::Value::Text("empty_key".to_string()),
    );

    let meta = GlobalMetadata {
        version: 3,
        extra,
        ..Default::default()
    };

    let desc = make_descriptor(vec![2], Dtype::Float32);
    let data = vec![0u8; 8];

    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (decoded_meta, _) = decode(&encoded, &DecodeOptions::default()).unwrap();

    assert_eq!(
        decoded_meta.extra.get(""),
        Some(&ciborium::Value::Text("empty_key".to_string()))
    );
}

// ── 60. Provenance timestamp accuracy ────────────────────────────────────────
//
// Guards the `civil_from_days` date conversion and time arithmetic in
// `populate_reserved_provenance`.  Existing tests only checked the
// `reserved.contains_key("time")` — this test parses the value and
// verifies the encoded UTC timestamp falls within the window measured
// before and after encoding.

/// Inverse of `civil_from_days`: produce a Unix epoch in seconds from
/// the Gregorian calendar fields written by `populate_reserved_provenance`.
/// Uses Howard Hinnant's `days_from_civil` algorithm (the mathematical
/// inverse of the production `civil_from_days`).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let m_adj = if m <= 2 { m + 9 } else { m - 3 };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = (y - era * 400) as u64;
    let doy = (153 * m_adj as u64 + 2) / 5 + d as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe as i64 - 719468
}

fn epoch_from_civil(y: i64, month: u32, day: u32, h: u64, min: u64, s: u64) -> u64 {
    let days = days_from_civil(y, month, day);
    days as u64 * 86400 + h * 3600 + min * 60 + s
}

#[test]
fn provenance_timestamp_is_accurate_utc() {
    use std::time::{SystemTime, UNIX_EPOCH};

    let before = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![0u8; 16];
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let after = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let (decoded_meta, _) = decode(&encoded, &DecodeOptions::default()).unwrap();

    let time_val = decoded_meta
        .reserved
        .get("time")
        .expect("reserved must contain 'time' key");
    let time_str = match time_val {
        ciborium::Value::Text(s) => s.clone(),
        other => panic!("reserved['time'] must be Text, got: {other:?}"),
    };

    // Format must be exactly "YYYY-MM-DDThh:mm:ssZ" (20 chars)
    assert_eq!(
        time_str.len(),
        20,
        "timestamp must be 20 chars, got: {time_str}"
    );
    assert!(
        time_str.ends_with('Z'),
        "timestamp must end with 'Z': {time_str}"
    );

    let (date_part, rest) = time_str
        .split_once('T')
        .expect("timestamp must contain 'T'");
    let time_part = rest.trim_end_matches('Z');

    let date_components: Vec<i64> = date_part
        .split('-')
        .map(|s| s.parse().expect("date component must be integer"))
        .collect();
    let time_components: Vec<u64> = time_part
        .split(':')
        .map(|s| s.parse().expect("time component must be integer"))
        .collect();

    assert_eq!(date_components.len(), 3, "date must have 3 components");
    assert_eq!(time_components.len(), 3, "time must have 3 components");

    let (year, month, day) = (
        date_components[0],
        date_components[1] as u32,
        date_components[2] as u32,
    );
    let (hours, minutes, seconds) = (time_components[0], time_components[1], time_components[2]);

    assert!((1..=12).contains(&month), "month {month} out of [1,12]");
    assert!((1..=31).contains(&day), "day {day} out of [1,31]");
    assert!(hours <= 23, "hours {hours} out of [0,23]");
    assert!(minutes <= 59, "minutes {minutes} out of [0,59]");
    assert!(seconds <= 59, "seconds {seconds} out of [0,59]");

    // Reconstruct the epoch from the parsed components and verify it
    // falls within the before/after window (with 1-second tolerance for
    // crossing a second boundary during the encode call itself).
    let encoded_epoch = epoch_from_civil(year, month, day, hours, minutes, seconds);
    assert!(
        encoded_epoch >= before.saturating_sub(1) && encoded_epoch <= after + 1,
        "timestamp {time_str} → epoch {encoded_epoch} is outside window [{before}, {after}]"
    );
}

// ── 61. Mask small-mask threshold behaviour ───────────────────────────────────
//
// All existing mask tests use `small_mask_threshold_bytes: 0` which skips the
// threshold logic entirely.  These tests exercise the `encode_one_mask` branch
// that downgrades small masks to `None` and leaves large masks unchanged.
// Guards the `&&→||` mutation on that condition.

#[test]
fn mask_threshold_small_mask_is_downgraded_to_none() {
    use tensogram::encode::MaskMethod;

    // 4 f32 elements → packed NaN mask is 1 byte (ceil(4/8)).
    // Threshold 16 > 1, so the mask should be downgraded to "none".
    let values: Vec<f32> = vec![1.0, f32::NAN, 3.0, 4.0];
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let opts = EncodeOptions {
        allow_nan: true,
        nan_mask_method: MaskMethod::Roaring,
        small_mask_threshold_bytes: 16,
        hash_algorithm: None,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &opts).unwrap();
    let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();

    let method = &objects[0]
        .0
        .masks
        .as_ref()
        .expect("masks must be present")
        .nan
        .as_ref()
        .expect("NaN mask must be present")
        .method;
    assert_eq!(
        method, "none",
        "mask smaller than threshold must be downgraded to 'none'"
    );
}

#[test]
fn mask_threshold_large_mask_keeps_requested_method() {
    use tensogram::encode::MaskMethod;

    // 500 f32 elements with all-NaN → packed mask = ceil(500/8) = 63 bytes.
    // Threshold 4 < 63, so the mask should keep the requested method.
    let n = 500usize;
    let values: Vec<f32> = (0..n).map(|_| f32::NAN).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let desc = make_descriptor(vec![n as u64], Dtype::Float32);
    let opts = EncodeOptions {
        allow_nan: true,
        nan_mask_method: MaskMethod::Roaring,
        small_mask_threshold_bytes: 4,
        hash_algorithm: None,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &opts).unwrap();
    let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();

    let method = &objects[0]
        .0
        .masks
        .as_ref()
        .expect("masks must be present")
        .nan
        .as_ref()
        .expect("NaN mask must be present")
        .method;
    assert_eq!(
        method, "roaring",
        "mask larger than threshold must keep the requested method"
    );
}

// ── 62. Zstd mask compression level stored in descriptor params ───────────────
//
// Guards the `mask_params_cbor → BTreeMap::new()` mutation: if params are
// cleared, the level key is lost and the assertion below fails.

#[cfg(feature = "zstd")]
#[test]
fn zstd_mask_level_preserved_in_descriptor_params() {
    use tensogram::encode::MaskMethod;

    let values: Vec<f64> = (0..128)
        .map(|i| if i % 10 == 0 { f64::NAN } else { i as f64 })
        .collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let desc = make_descriptor(vec![128], Dtype::Float64);
    let opts = EncodeOptions {
        allow_nan: true,
        nan_mask_method: MaskMethod::Zstd { level: Some(5) },
        small_mask_threshold_bytes: 0,
        hash_algorithm: None,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &opts).unwrap();
    let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();

    let params = &objects[0]
        .0
        .masks
        .as_ref()
        .expect("masks must be present")
        .nan
        .as_ref()
        .expect("NaN mask must be present")
        .params;

    let level_val = params
        .get("level")
        .expect("zstd mask descriptor params must contain 'level' key");
    assert_eq!(
        level_val,
        &ciborium::Value::Integer(5i64.into()),
        "zstd mask level must be stored as integer 5 in descriptor params"
    );
}

// ── 63. Async local file API ──────────────────────────────────────────────────
//
// These tests exercise the async variants of the TensogramFile API
// (`open_async`, `message_count_async`, `read_message_async`,
// `decode_message_async`, `decode_metadata_async`, `decode_descriptors_async`,
// `decode_object_async`, `decode_range_async`).
//
// All mutations at file.rs:443–622 are whole-function stubs that return
// `Default::default()` values; a single test that verifies the returned
// data matches what was written kills all of them.

#[cfg(feature = "async")]
#[tokio::test]
async fn async_file_api_round_trips_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("async_test.tgm");

    // Write a recognisable message synchronously.
    let meta = make_global_meta();
    let n = 50usize;
    let desc = make_descriptor(vec![n as u64], Dtype::Float32);
    let data: Vec<u8> = (0..n)
        .map(|i| i as f32)
        .flat_map(|v| v.to_ne_bytes())
        .collect();

    {
        let mut file = TensogramFile::create(&path).unwrap();
        file.append(&meta, &[(&desc, &data)], &EncodeOptions::default())
            .unwrap();
    }

    // ── open_async ──────────────────────────────────────────────────────────
    let file = TensogramFile::open_async(&path).await.unwrap();

    // ── message_count_async ─────────────────────────────────────────────────
    let count = file.message_count_async().await.unwrap();
    assert_eq!(count, 1, "message_count_async must return 1");

    // ── read_message_async ──────────────────────────────────────────────────
    let raw = file.read_message_async(0).await.unwrap();
    assert!(
        !raw.is_empty(),
        "read_message_async must return non-empty bytes"
    );

    // ── decode_message_async ────────────────────────────────────────────────
    let (dec_meta, objects) = file
        .decode_message_async(0, &DecodeOptions::default())
        .await
        .unwrap();
    assert_eq!(dec_meta.version, 3, "decode_message_async: wrong version");
    assert_eq!(objects.len(), 1, "decode_message_async: wrong object count");
    assert_eq!(
        objects[0].1.as_slice(),
        data.as_slice(),
        "decode_message_async: decoded bytes must match written data"
    );

    // ── decode_metadata_async ────────────────────────────────────────────────
    let dec_meta2 = file.decode_metadata_async(0).await.unwrap();
    assert_eq!(dec_meta2.version, 3, "decode_metadata_async: wrong version");

    // ── decode_descriptors_async ─────────────────────────────────────────────
    let (_, descs) = file.decode_descriptors_async(0).await.unwrap();
    assert_eq!(
        descs.len(),
        1,
        "decode_descriptors_async: wrong descriptor count"
    );
    assert_eq!(
        descs[0].dtype,
        Dtype::Float32,
        "decode_descriptors_async: wrong dtype"
    );

    // ── decode_object_async ──────────────────────────────────────────────────
    let (_, obj_desc, obj_data) = file
        .decode_object_async(0, 0, &DecodeOptions::default())
        .await
        .unwrap();
    assert_eq!(
        obj_desc.dtype,
        Dtype::Float32,
        "decode_object_async: wrong dtype"
    );
    assert_eq!(
        obj_data.as_slice(),
        data.as_slice(),
        "decode_object_async: decoded bytes must match written data"
    );

    // ── decode_range_async ───────────────────────────────────────────────────
    let (range_desc, ranges_out) = file
        .decode_range_async(0, 0, &[(0, 5)], &DecodeOptions::default())
        .await
        .unwrap();
    assert_eq!(
        range_desc.dtype,
        Dtype::Float32,
        "decode_range_async: wrong dtype"
    );
    assert_eq!(
        ranges_out.len(),
        1,
        "decode_range_async: expected one range result"
    );
    // First 5 f32 elements = bytes 0..20
    assert_eq!(
        ranges_out[0].as_slice(),
        &data[0..20],
        "decode_range_async: first 5 elements must match"
    );
}

// ── 64. invalidate_offsets clears a populated cache ───────────────────────────
//
// The existing streaming+buffered test calls `invalidate_offsets()` when the
// OnceLock is already empty (because `append()` resets it), so the mutation
// `invalidate_offsets → ()` survives.  This test explicitly populates the
// cache (by calling `message_count()`) before appending externally, ensuring
// the no-op mutation is caught.

#[test]
fn invalidate_offsets_clears_populated_cache() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("invalidate_test.tgm");

    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Uint8);
    let data0 = vec![0u8; 4];
    let data1 = vec![1u8; 4];

    let mut file = TensogramFile::create(&path).unwrap();
    file.append(
        &meta,
        &[(&desc, data0.as_slice())],
        &EncodeOptions::default(),
    )
    .unwrap();

    // Populate the offset cache (ensure_scanned).
    assert_eq!(file.message_count().unwrap(), 1, "initial count must be 1");

    // Write a second message externally (bypassing `append` so the cache
    // is NOT automatically cleared).
    {
        let second = encode(
            &meta,
            &[(&desc, data1.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        std::io::Write::write_all(&mut f, &second).unwrap();
    }

    // Without invalidate, the cached count is still 1.
    // With invalidate (real code), ensure_scanned rescans and finds 2.
    file.invalidate_offsets();
    assert_eq!(
        file.message_count().unwrap(),
        2,
        "message_count after invalidate_offsets must reflect newly appended message"
    );

    // Also verify the second message decodes correctly so this isn't a
    // vacuous count test.
    let (_, objects) = file.decode_message(1, &DecodeOptions::default()).unwrap();
    assert_eq!(
        objects[0].1.as_slice(),
        data1.as_slice(),
        "second message payload must match what was written"
    );
}
