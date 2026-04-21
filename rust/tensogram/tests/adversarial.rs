// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::collections::BTreeMap;
use tensogram::*;

fn make_simple_float32_pair(shape: Vec<u64>) -> (GlobalMetadata, DataObjectDescriptor) {
    let strides: Vec<u64> = if shape.is_empty() {
        vec![]
    } else {
        let mut s = vec![1u64; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            s[i] = s[i + 1] * shape[i + 1];
        }
        s
    };
    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: shape.len() as u64,
        shape,
        strides,
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
        hash: None,
    };
    (global, desc)
}

fn make_shuffle_pair(shape: Vec<u64>, element_size: u64) -> (GlobalMetadata, DataObjectDescriptor) {
    let strides: Vec<u64> = if shape.is_empty() {
        vec![]
    } else {
        let mut s = vec![1u64; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            s[i] = s[i + 1] * shape[i + 1];
        }
        s
    };
    let mut params = BTreeMap::new();
    params.insert(
        "shuffle_element_size".to_string(),
        ciborium::Value::Integer(element_size.into()),
    );
    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: shape.len() as u64,
        shape,
        strides,
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "shuffle".to_string(),
        compression: "none".to_string(),
        params,
        masks: None,
        hash: None,
    };
    (global, desc)
}

// --- Adversarial wire-level tests ---
// The v1 BinaryHeader / TERMINATOR / OBJS / OBJE markers no longer exist in v2.
// These tests exercise the same validation behaviors using truncated or corrupted
// v2 buffers produced by the normal encoder.

#[test]
fn test_adversarial_truncated_message_rejected() {
    // A legitimately encoded message, truncated to half its length, must be rejected.
    let (global, desc) = make_simple_float32_pair(vec![4]);
    let data = vec![0u8; 4 * 4];
    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let truncated = &encoded[..encoded.len() / 2];
    let result = decode(truncated, &DecodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for truncated message but got Ok: {:?}",
        result
    );
}

#[test]
fn test_adversarial_wrong_magic_rejected() {
    // Overwrite the first 8 bytes (magic) to produce invalid magic.
    let (global, desc) = make_simple_float32_pair(vec![4]);
    let data = vec![0u8; 4 * 4];
    let mut encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    encoded[0..8].copy_from_slice(b"BADMAGIC");
    let result = decode(&encoded, &DecodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for wrong magic but got Ok: {:?}",
        result
    );
}

#[test]
fn test_adversarial_corrupted_end_magic_rejected() {
    // Overwrite the last 8 bytes (end magic) to trigger postamble validation failure.
    let (global, desc) = make_simple_float32_pair(vec![4]);
    let data = vec![0u8; 4 * 4];
    let mut encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let len = encoded.len();
    encoded[len - 8..].copy_from_slice(b"BADMAGIC");
    let result = decode(&encoded, &DecodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for corrupted end magic but got Ok: {:?}",
        result
    );
}

#[test]
fn test_adversarial_empty_buffer_rejected() {
    let result = decode(&[], &DecodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for empty buffer but got Ok: {:?}",
        result
    );
}

#[test]
fn test_adversarial_negative_cbor_int_wraps() {
    let below_i32_min: i64 = i32::MIN as i64 - 1;
    let mut params = BTreeMap::new();
    params.insert("reference_value".to_string(), ciborium::Value::Float(0.0));
    params.insert(
        "binary_scale_factor".to_string(),
        ciborium::Value::Integer(below_i32_min.into()),
    );
    params.insert(
        "decimal_scale_factor".to_string(),
        ciborium::Value::Integer(0.into()),
    );
    params.insert(
        "bits_per_value".to_string(),
        ciborium::Value::Integer(16.into()),
    );

    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
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
        hash: None,
    };

    let data = vec![0u8; 4 * 8];
    let result = encode(&global, &[(&desc, &data)], &EncodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for out-of-range binary_scale_factor but got Ok: {:?}",
        result
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("binary_scale_factor"),
        "expected 'binary_scale_factor' in error message, got: {msg}"
    );
}

#[test]
fn test_adversarial_non_f64_simple_packing() {
    let mut params = BTreeMap::new();
    params.insert("reference_value".to_string(), ciborium::Value::Float(0.0));
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

    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![10],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params,
        masks: None,
        hash: None,
    };

    let data = vec![0u8; 10 * 4];
    let result = encode(&global, &[(&desc, &data)], &EncodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for simple_packing with Float32 but got Ok: {:?}",
        result
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("simple_packing") || msg.contains("float64") || msg.contains("f64"),
        "expected 'simple_packing' or 'float64' in error message, got: {msg}"
    );
}

#[test]
fn test_adversarial_shuffle_element_size_zero() {
    let (global, desc) = make_shuffle_pair(vec![10], 0);
    let data = vec![0u8; 10 * 4];
    let result = encode(&global, &[(&desc, &data)], &EncodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for shuffle_element_size=0 but got Ok: {:?}",
        result
    );
}

#[test]
fn test_adversarial_shuffle_misaligned() {
    let float32_byte_width: usize = 4;
    let num_elements: usize = 10;
    let element_size_that_doesnt_divide_40: u64 = 3;

    let (global, desc) = make_shuffle_pair(
        vec![num_elements as u64],
        element_size_that_doesnt_divide_40,
    );
    let data = vec![0u8; num_elements * float32_byte_width];
    let result = encode(&global, &[(&desc, &data)], &EncodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for misaligned shuffle (element_size=3 on 40-byte Float32 data) but got Ok: {:?}",
        result
    );
}

#[test]
fn test_adversarial_decode_range_with_shuffle() {
    let float32_element_size: u64 = 4;
    let (global, desc) = make_shuffle_pair(vec![10], float32_element_size);
    let data: Vec<u8> = (0u8..40).collect();

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default())
        .expect("encode with shuffle must succeed for this test");

    let result = decode_range(&encoded, 0, &[(0, 5)], &DecodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for decode_range on shuffled payload but got Ok: {:?}",
        result
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("shuffle") || msg.contains("filter"),
        "expected 'shuffle' or 'filter' in error message, got: {msg}"
    );
}

#[test]
fn test_adversarial_shape_product_overflow() {
    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 2,
        shape: vec![u64::MAX, 2],
        strides: vec![2, 1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
        hash: None,
    };

    let data = vec![0u8; 64];
    let result = encode(&global, &[(&desc, &data)], &EncodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for shape product overflow but got Ok: {:?}",
        result
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("overflow"),
        "expected 'overflow' in error message, got: {msg}"
    );
}

#[test]
fn test_adversarial_empty_obj_type() {
    let (global, mut desc) = make_simple_float32_pair(vec![4]);
    desc.obj_type = String::new();

    let data = vec![0u8; 4 * 4];
    let result = encode(&global, &[(&desc, &data)], &EncodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for empty obj_type but got Ok: {:?}",
        result
    );
}

#[test]
fn test_adversarial_ndim_mismatch() {
    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 5,
        shape: vec![4, 5],
        strides: vec![5, 1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
        hash: None,
    };

    let data = vec![0u8; 4 * 5 * 4];
    let result = encode(&global, &[(&desc, &data)], &EncodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for ndim/shape mismatch but got Ok: {:?}",
        result
    );
}
