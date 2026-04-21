// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::collections::BTreeMap;
use tensogram::*;
use tensogram_encodings::simple_packing;

fn make_float32_descriptor(shape: Vec<u64>) -> (GlobalMetadata, DataObjectDescriptor) {
    let strides = compute_strides(&shape);
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
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    (global, desc)
}

fn make_mars_pair(shape: Vec<u64>, param: &str) -> (GlobalMetadata, DataObjectDescriptor) {
    let strides = compute_strides(&shape);

    let mut mars_global = BTreeMap::new();
    mars_global.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
    mars_global.insert("type".to_string(), ciborium::Value::Text("fc".to_string()));
    mars_global.insert(
        "date".to_string(),
        ciborium::Value::Text("20260401".to_string()),
    );

    let mut base_entry = BTreeMap::new();
    base_entry.insert(
        "mars".to_string(),
        ciborium::Value::Map(
            mars_global
                .into_iter()
                .map(|(k, v)| (ciborium::Value::Text(k), v))
                .collect(),
        ),
    );

    let global = GlobalMetadata {
        version: 3,
        base: vec![base_entry],
        ..Default::default()
    };

    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: shape.len() as u64,
        shape,
        strides,
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: {
            let mut p = BTreeMap::new();
            p.insert(
                "mars_param".to_string(),
                ciborium::Value::Text(param.to_string()),
            );
            p
        },
        masks: None,
    };
    (global, desc)
}

fn compute_strides(shape: &[u64]) -> Vec<u64> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1u64; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[test]
fn test_full_round_trip_single_object() {
    let (global, desc) = make_float32_descriptor(vec![10, 20]);
    let data = vec![0u8; 10 * 20 * 4]; // 200 float32 = 800 bytes

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Verify magic and terminator
    assert_eq!(&encoded[0..8], b"TENSOGRM");
    assert_eq!(&encoded[encoded.len() - 8..], b"39277777");

    let (decoded_meta, decoded_objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded_meta.version, 3);
    assert_eq!(decoded_objects.len(), 1);
    assert_eq!(decoded_objects[0].1, data);
}

#[test]
fn test_multi_object_message() {
    let strides1 = compute_strides(&[4, 5]);

    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };

    let desc1 = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 2,
        shape: vec![4, 5],
        strides: strides1,
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    let desc2 = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![3],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };

    let data1 = vec![1u8; 4 * 5 * 4]; // 20 float32
    let data2 = vec![2u8; 3 * 8]; // 3 float64

    let encoded = encode(
        &global,
        &[(&desc1, &data1), (&desc2, &data2)],
        &EncodeOptions::default(),
    )
    .unwrap();
    let (meta, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();

    assert_eq!(objects.len(), 2);
    assert_eq!(objects[0].1, data1);
    assert_eq!(objects[1].1, data2);
    let _ = meta;
}

#[test]
fn test_decode_metadata_only() {
    let (global, desc) = make_mars_pair(vec![10], "2t");
    let data = vec![0u8; 10 * 4];

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let meta = decode_metadata(&encoded).unwrap();

    assert_eq!(meta.version, 3);
}

#[test]
fn test_decode_single_object_by_index() {
    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };

    let desc1 = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![2],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    let desc2 = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![3],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };

    let data1 = vec![0xAA; 2 * 4];
    let data2 = vec![0xBB; 3 * 4];

    let encoded = encode(
        &global,
        &[(&desc1, &data1), (&desc2, &data2)],
        &EncodeOptions::default(),
    )
    .unwrap();

    let (_, _returned_desc, obj) = decode_object(&encoded, 1, &DecodeOptions::default()).unwrap();
    assert_eq!(obj, data2);
}

#[test]
fn test_zero_object_message() {
    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };

    let encoded = encode(&global, &[], &EncodeOptions::default()).unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 0);
}

#[test]
fn test_hash_verification_passes() {
    let (global, desc) = make_float32_descriptor(vec![4]);
    let data = vec![42u8; 4 * 4];

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Decode with hash verification enabled
    let options = DecodeOptions {
        verify_hash: true,
        ..Default::default()
    };
    let (_, objects) = decode(&encoded, &options).unwrap();
    assert_eq!(objects[0].1, data);
}

/// Flipping a byte inside the payload region of a hashed message
/// must surface as a `HashMismatch` validation issue.  v3 moved
/// frame-level integrity from the decoder to the validator
/// (see `plans/WIRE_FORMAT.md` §11.1): `DecodeOptions.verify_hash`
/// is a no-op at the decode layer; frame-body hashes are
/// recomputed by `validate::validate_message` at Integrity level.
#[test]
fn test_hash_verification_fails_on_corruption() {
    use tensogram::validate::{IssueCode, ValidateOptions, validate_message};

    let (global, desc) = make_float32_descriptor(vec![100]);
    let data = vec![42u8; 100 * 4];
    let mut encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Identify the NTensorFrame by its "FR" magic + type=9 bytes.
    let data_frame_marker: &[u8] = &[b'F', b'R', 0x00, 0x09];
    let frame_start = encoded
        .windows(4)
        .position(|w| w == data_frame_marker)
        .expect("NTensorFrame not found in encoded message");
    // Flip a byte 16 B past the frame header — inside the payload.
    encoded[frame_start + 16] ^= 0xFF;

    let report = validate_message(&encoded, &ValidateOptions::default());
    assert!(
        report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::HashMismatch),
        "expected HashMismatch after payload tamper, got: {:?}",
        report.issues
    );
}

#[test]
fn test_simple_packing_round_trip() {
    let values: Vec<f64> = (0..100).map(|i| 250.0 + i as f64 * 0.1).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let params = tensogram_encodings::simple_packing::compute_params(&values, 16, 0).unwrap();

    let mut packing_params = BTreeMap::new();
    packing_params.insert(
        "reference_value".to_string(),
        ciborium::Value::Float(params.reference_value),
    );
    packing_params.insert(
        "binary_scale_factor".to_string(),
        ciborium::Value::Integer((params.binary_scale_factor as i64).into()),
    );
    packing_params.insert(
        "decimal_scale_factor".to_string(),
        ciborium::Value::Integer((params.decimal_scale_factor as i64).into()),
    );
    packing_params.insert(
        "bits_per_value".to_string(),
        ciborium::Value::Integer((params.bits_per_value as i64).into()),
    );

    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![100],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::native(),
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: packing_params,
        masks: None,
    };

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();

    // Decoded values should be f64 bytes
    let decoded_values: Vec<f64> = objects[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(decoded_values.len(), 100);
    for (orig, dec) in values.iter().zip(decoded_values.iter()) {
        assert!((orig - dec).abs() < 0.01, "orig={orig}, dec={dec}");
    }
}

#[test]
fn test_shuffle_round_trip() {
    // 10 float32 values with shuffle filter
    let data: Vec<u8> = (0..40).collect(); // 10 * 4 bytes

    let mut params = BTreeMap::new();
    params.insert(
        "shuffle_element_size".to_string(),
        ciborium::Value::Integer(4.into()),
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
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "shuffle".to_string(),
        compression: "none".to_string(),
        params,
        masks: None,
    };

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(objects[0].1, data);
}

#[test]
fn test_scan_multi_message_buffer() {
    let (global1, desc1) = make_mars_pair(vec![4], "2t");
    let (global2, desc2) = make_mars_pair(vec![8], "10u");
    let data1 = vec![0u8; 4 * 4];
    let data2 = vec![0u8; 8 * 4];

    let msg1 = encode(&global1, &[(&desc1, &data1)], &EncodeOptions::default()).unwrap();
    let msg2 = encode(&global2, &[(&desc2, &data2)], &EncodeOptions::default()).unwrap();

    let mut buf = Vec::new();
    buf.extend_from_slice(&msg1);
    buf.extend_from_slice(&msg2);

    let offsets = scan(&buf);
    assert_eq!(offsets.len(), 2);

    // Decode each message from the scanned offsets
    let (_, objects1) = decode(
        &buf[offsets[0].0..offsets[0].0 + offsets[0].1],
        &DecodeOptions::default(),
    )
    .unwrap();
    let (_, objects2) = decode(
        &buf[offsets[1].0..offsets[1].0 + offsets[1].1],
        &DecodeOptions::default(),
    )
    .unwrap();
    assert_eq!(objects1[0].0.shape, vec![4]);
    assert_eq!(objects2[0].0.shape, vec![8]);
}

#[test]
fn test_partial_range_decode_uncompressed() {
    // 10 float32 values, decode elements 3..6
    let values: Vec<f32> = (0..10).map(|i| i as f32 * 1.5).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let (global, desc) = make_float32_descriptor(vec![10]);

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Decode range: elements 3..6 (3 elements)
    let (_, partial) =
        decode_range(&encoded, 0, &[(3, 3)], &DecodeOptions::default()).expect("decode_range");

    // One range requested → one result part
    assert_eq!(partial.len(), 1, "expected 1 part for 1 range");

    let expected: Vec<u8> = values[3..6].iter().flat_map(|v| v.to_ne_bytes()).collect();
    assert_eq!(partial[0], expected);

    // Also verify join produces the same result
    let joined: Vec<u8> = partial.into_iter().flatten().collect();
    assert_eq!(joined, expected);
}

#[test]
fn test_decode_range_shuffle_rejected() {
    let data: Vec<u8> = (0..40).collect();

    let mut params = BTreeMap::new();
    params.insert(
        "shuffle_element_size".to_string(),
        ciborium::Value::Integer(4.into()),
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
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "shuffle".to_string(),
        compression: "none".to_string(),
        params,
        masks: None,
    };

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let err = decode_range(&encoded, 0, &[(3, 3)], &DecodeOptions::default()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("shuffle") || msg.contains("filter"), "{msg}");
}

#[test]
fn test_file_multi_message_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi.tgm");

    let mut file = TensogramFile::create(&path).unwrap();

    // Append 3 messages with different data
    for i in 0..3u8 {
        let (global, desc) = make_float32_descriptor(vec![4]);
        let data = vec![i; 4 * 4];
        file.append(&global, &[(&desc, &data)], &EncodeOptions::default())
            .unwrap();
    }

    assert_eq!(file.message_count().unwrap(), 3);

    // Read back and verify each
    for i in 0..3u8 {
        let (_, objects) = file
            .decode_message(i as usize, &DecodeOptions::default())
            .unwrap();
        assert_eq!(objects[0].1, vec![i; 4 * 4]);
    }
}

#[test]
fn test_namespaced_metadata_round_trip() {
    let (global, desc) = make_mars_pair(vec![4], "wave_spectra");
    let data = vec![0u8; 4 * 4];

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let meta = decode_metadata(&encoded).unwrap();

    // Verify mars namespace in base[0] metadata
    assert!(meta.base[0].contains_key("mars"));
    if let ciborium::Value::Map(entries) = &meta.base[0]["mars"] {
        let class_val = entries
            .iter()
            .find(|(k, _)| matches!(k, ciborium::Value::Text(s) if s == "class"))
            .map(|(_, v)| v);
        assert!(matches!(class_val, Some(ciborium::Value::Text(s)) if s == "od"));
    }
}

#[test]
fn test_validate_object_overflow() {
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
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };

    let data = vec![0u8; 64];
    let result = encode(&global, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_err(), "expected Err but got Ok");
}

#[test]
fn test_cross_endian_round_trip() {
    // Encode the same float values with different wire byte orders.
    // With native_byte_order=true (default), both should decode to
    // identical native-endian bytes.
    let values: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.5).collect();

    let params = tensogram_encodings::simple_packing::compute_params(&values, 16, 0).unwrap();

    let mut packing_params = BTreeMap::new();
    packing_params.insert(
        "reference_value".to_string(),
        ciborium::Value::Float(params.reference_value),
    );
    packing_params.insert(
        "binary_scale_factor".to_string(),
        ciborium::Value::Integer((params.binary_scale_factor as i64).into()),
    );
    packing_params.insert(
        "decimal_scale_factor".to_string(),
        ciborium::Value::Integer((params.decimal_scale_factor as i64).into()),
    );
    packing_params.insert(
        "bits_per_value".to_string(),
        ciborium::Value::Integer((params.bits_per_value as i64).into()),
    );

    let be_data: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();

    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };
    let be_desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![50],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Big,
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: packing_params.clone(),
        masks: None,
    };

    let encoded_be = encode(&global, &[(&be_desc, &be_data)], &EncodeOptions::default()).unwrap();

    // Default decode: native byte order
    let (_, objects_be) = decode(&encoded_be, &DecodeOptions::default()).unwrap();
    let decoded_be_values: Vec<f64> = objects_be[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    for (orig, dec) in values.iter().zip(decoded_be_values.iter()) {
        assert!(
            (orig - dec).abs() < 0.01,
            "BE→native round-trip: orig={orig}, dec={dec}"
        );
    }

    let le_data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let le_desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![50],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Little,
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: packing_params.clone(),
        masks: None,
    };

    let encoded_le = encode(&global, &[(&le_desc, &le_data)], &EncodeOptions::default()).unwrap();
    let (_, objects_le) = decode(&encoded_le, &DecodeOptions::default()).unwrap();
    let decoded_le_values: Vec<f64> = objects_le[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    for (orig, dec) in values.iter().zip(decoded_le_values.iter()) {
        assert!(
            (orig - dec).abs() < 0.01,
            "LE→native round-trip: orig={orig}, dec={dec}"
        );
    }

    // Both decode to native byte order → decoded bytes should be equal.
    assert_eq!(
        objects_be[0].1, objects_le[0].1,
        "BE and LE should both decode to identical native-endian bytes"
    );

    // With native_byte_order=false, the wire byte orders are preserved,
    // so the two outputs should differ.
    let wire_opts = DecodeOptions {
        native_byte_order: false,
        ..Default::default()
    };
    let (_, wire_be) = decode(&encoded_be, &wire_opts).unwrap();
    let (_, wire_le) = decode(&encoded_le, &wire_opts).unwrap();
    assert_ne!(
        wire_be[0].1, wire_le[0].1,
        "wire-order BE and LE bytes should differ"
    );
}

// ── decode_range with cross-endian + native_byte_order ───────────────────────

#[test]
fn test_decode_range_cross_endian_native() {
    // Encode float32 data with byte_order=Big, then decode_range with
    // native_byte_order=true (default).  The returned bytes should be
    // in native byte order.
    let values: Vec<f32> = (0..20).map(|i| i as f32 * 1.5).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();

    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![20],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // decode_range elements 5..10 (5 elements)
    let (_, parts) =
        decode_range(&encoded, 0, &[(5, 5)], &DecodeOptions::default()).expect("decode_range");
    let part_values: Vec<f32> = parts[0]
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(part_values.len(), 5);
    for (orig, dec) in values[5..10].iter().zip(part_values.iter()) {
        assert_eq!(*orig, *dec, "cross-endian decode_range mismatch");
    }
}

#[test]
fn test_decode_range_wire_byte_order_opt_out() {
    // decode_range with native_byte_order=false should return wire-order bytes.
    let values: Vec<f32> = (0..20).map(|i| i as f32 * 1.5).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();

    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![20],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let wire_opts = DecodeOptions {
        native_byte_order: false,
        ..Default::default()
    };
    let (_, parts) = decode_range(&encoded, 0, &[(5, 5)], &wire_opts).expect("decode_range");
    // Wire bytes should be big-endian
    let wire_values: Vec<f32> = parts[0]
        .chunks_exact(4)
        .map(|c| f32::from_be_bytes(c.try_into().unwrap()))
        .collect();
    for (orig, dec) in values[5..10].iter().zip(wire_values.iter()) {
        assert_eq!(*orig, *dec, "wire-order decode_range mismatch");
    }
}

#[test]
fn test_simple_packing_rejects_non_f64() {
    let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let params = tensogram_encodings::simple_packing::compute_params(&values, 16, 0).unwrap();

    let mut packing_params = BTreeMap::new();
    packing_params.insert(
        "reference_value".to_string(),
        ciborium::Value::Float(params.reference_value),
    );
    packing_params.insert(
        "binary_scale_factor".to_string(),
        ciborium::Value::Integer((params.binary_scale_factor as i64).into()),
    );
    packing_params.insert(
        "decimal_scale_factor".to_string(),
        ciborium::Value::Integer((params.decimal_scale_factor as i64).into()),
    );
    packing_params.insert(
        "bits_per_value".to_string(),
        ciborium::Value::Integer((params.bits_per_value as i64).into()),
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
        byte_order: ByteOrder::native(),
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: packing_params,
        masks: None,
    };

    let data = vec![0u8; 10 * 4];
    let result = encode(&global, &[(&desc, &data)], &EncodeOptions::default());
    assert!(
        result.is_err(),
        "expected error for simple_packing with Float32 dtype"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("float64") || msg.contains("f64"),
        "expected 'float64' in error, got: {msg}"
    );
}

#[test]
fn test_validate_ndim_mismatch() {
    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 3,
        shape: vec![4, 5],
        strides: vec![5, 1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };

    let data = vec![0u8; 4 * 5 * 4];
    let result = encode(&global, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_err(), "expected Err but got Ok");
}

#[test]
fn test_param_out_of_bounds() {
    let mut packing_params = BTreeMap::new();
    packing_params.insert("reference_value".to_string(), ciborium::Value::Float(0.0));
    packing_params.insert(
        "binary_scale_factor".to_string(),
        ciborium::Value::Integer(i64::MAX.into()),
    );
    packing_params.insert(
        "decimal_scale_factor".to_string(),
        ciborium::Value::Integer(0.into()),
    );
    packing_params.insert(
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
        byte_order: ByteOrder::native(),
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: packing_params,
        masks: None,
    };

    let data = vec![0u8; 4 * 8];
    let result = encode(&global, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_err(), "expected Err but got Ok");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("binary_scale_factor"),
        "expected 'binary_scale_factor' in error, got: {msg}"
    );
}

// --- szip compression integration tests ---

/// Build (GlobalMetadata, DataObjectDescriptor) for simple_packing + szip pipeline.
fn make_szip_packing_pair(
    num_values: u64,
    params: &simple_packing::SimplePackingParams,
) -> (GlobalMetadata, DataObjectDescriptor) {
    let mut packing_params = BTreeMap::new();
    packing_params.insert(
        "reference_value".to_string(),
        ciborium::Value::Float(params.reference_value),
    );
    packing_params.insert(
        "binary_scale_factor".to_string(),
        ciborium::Value::Integer((params.binary_scale_factor as i64).into()),
    );
    packing_params.insert(
        "decimal_scale_factor".to_string(),
        ciborium::Value::Integer((params.decimal_scale_factor as i64).into()),
    );
    packing_params.insert(
        "bits_per_value".to_string(),
        ciborium::Value::Integer((params.bits_per_value as i64).into()),
    );
    packing_params.insert("szip_rsi".to_string(), ciborium::Value::Integer(128.into()));
    packing_params.insert(
        "szip_block_size".to_string(),
        ciborium::Value::Integer(16.into()),
    );
    packing_params.insert(
        "szip_flags".to_string(),
        ciborium::Value::Integer(8_i64.into()),
    );

    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![num_values],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::native(),
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "szip".to_string(),
        params: packing_params,
        masks: None,
    };
    (global, desc)
}

/// Build (GlobalMetadata, DataObjectDescriptor) for raw szip compression (no encoding, no filter).
fn make_szip_raw_pair(num_values: u64, dtype: Dtype) -> (GlobalMetadata, DataObjectDescriptor) {
    let mut params = BTreeMap::new();
    params.insert("szip_rsi".to_string(), ciborium::Value::Integer(128.into()));
    params.insert(
        "szip_block_size".to_string(),
        ciborium::Value::Integer(16.into()),
    );
    params.insert(
        "szip_flags".to_string(),
        ciborium::Value::Integer(8_i64.into()),
    );

    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![num_values],
        strides: vec![1],
        dtype,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "szip".to_string(),
        params,
        masks: None,
    };
    (global, desc)
}

#[test]
fn test_szip_simple_packing_round_trip() {
    let values: Vec<f64> = (0..4096).map(|i| 250.0 + i as f64 * 0.1).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let packing = simple_packing::compute_params(&values, 16, 0).unwrap();
    let (global, desc) = make_szip_packing_pair(4096, &packing);

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Verify szip_block_offsets were stored in metadata
    let (_, objects_meta) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert!(objects_meta[0].0.params.contains_key("szip_block_offsets"));

    let decoded_values: Vec<f64> = objects_meta[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(decoded_values.len(), 4096);
    for (orig, dec) in values.iter().zip(decoded_values.iter()) {
        assert!((orig - dec).abs() < 0.01, "orig={orig}, dec={dec}");
    }
}

#[test]
fn test_szip_simple_packing_decode_range_vs_full() {
    let values: Vec<f64> = (0..4096).map(|i| 100.0 + i as f64 * 0.5).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let packing = simple_packing::compute_params(&values, 16, 0).unwrap();
    let (global, desc) = make_szip_packing_pair(4096, &packing);

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Full decode
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    let full_values: Vec<f64> = objects[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();

    // Partial decode: elements 100..600 (500 elements)
    let (_, partial_parts) =
        decode_range(&encoded, 0, &[(100, 500)], &DecodeOptions::default()).expect("decode_range");

    // One range requested → one result part
    assert_eq!(partial_parts.len(), 1, "expected 1 part for 1 range");

    let partial_bytes: Vec<u8> = partial_parts.into_iter().flatten().collect();
    let partial_values: Vec<f64> = partial_bytes
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(partial_values.len(), 500);
    for (full, partial) in full_values[100..600].iter().zip(partial_values.iter()) {
        assert!(
            (full - partial).abs() < 1e-10,
            "full={full}, partial={partial}"
        );
    }
}

#[test]
fn test_szip_simple_packing_decode_range_first_elements() {
    let values: Vec<f64> = (0..4096).map(|i| i as f64).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let packing = simple_packing::compute_params(&values, 16, 0).unwrap();
    let (global, desc) = make_szip_packing_pair(4096, &packing);

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    let full_values: Vec<f64> = objects[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();

    // First 10 elements
    let (_, partial_parts) =
        decode_range(&encoded, 0, &[(0, 10)], &DecodeOptions::default()).expect("decode_range");
    assert_eq!(partial_parts.len(), 1, "expected 1 part for 1 range");

    let partial_bytes: Vec<u8> = partial_parts.into_iter().flatten().collect();
    let partial_values: Vec<f64> = partial_bytes
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(partial_values.len(), 10);
    for (full, partial) in full_values[..10].iter().zip(partial_values.iter()) {
        assert!(
            (full - partial).abs() < 1e-10,
            "full={full}, partial={partial}"
        );
    }
}

#[test]
fn test_szip_simple_packing_decode_range_last_elements() {
    let values: Vec<f64> = (0..4096).map(|i| i as f64 * 3.125).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let packing = simple_packing::compute_params(&values, 16, 0).unwrap();
    let (global, desc) = make_szip_packing_pair(4096, &packing);

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    let full_values: Vec<f64> = objects[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();

    // Last 50 elements
    let (_, partial_parts) =
        decode_range(&encoded, 0, &[(4046, 50)], &DecodeOptions::default()).expect("decode_range");
    assert_eq!(partial_parts.len(), 1, "expected 1 part for 1 range");

    let partial_bytes: Vec<u8> = partial_parts.into_iter().flatten().collect();
    let partial_values: Vec<f64> = partial_bytes
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(partial_values.len(), 50);
    for (full, partial) in full_values[4046..].iter().zip(partial_values.iter()) {
        assert!(
            (full - partial).abs() < 1e-10,
            "full={full}, partial={partial}"
        );
    }
}

#[test]
fn test_szip_raw_u8_round_trip() {
    let data: Vec<u8> = (0..1024).flat_map(|i| (i as f32).to_ne_bytes()).collect();
    let (global, desc) = make_szip_raw_pair(4096, Dtype::Uint8);

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(objects[0].1, data);
}

#[test]
fn test_szip_shuffle_round_trip() {
    // shuffle + szip: float32 data shuffled to bytes, then szip-compressed
    let data: Vec<u8> = (0..1024).flat_map(|i| (i as f32).to_ne_bytes()).collect(); // 1024 finite f32s
    let mut params = BTreeMap::new();
    params.insert(
        "shuffle_element_size".to_string(),
        ciborium::Value::Integer(4.into()),
    );
    params.insert("szip_rsi".to_string(), ciborium::Value::Integer(128.into()));
    params.insert(
        "szip_block_size".to_string(),
        ciborium::Value::Integer(16.into()),
    );
    params.insert(
        "szip_flags".to_string(),
        ciborium::Value::Integer(8_i64.into()),
    );

    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![1024],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "shuffle".to_string(),
        compression: "szip".to_string(),
        params,
        masks: None,
    };

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(objects[0].1, data);
}

#[test]
fn test_szip_shuffle_decode_range_rejected() {
    // shuffle + szip: decode_range should be rejected
    let data: Vec<u8> = (0..1024).flat_map(|i| (i as f32).to_ne_bytes()).collect();
    let mut params = BTreeMap::new();
    params.insert(
        "shuffle_element_size".to_string(),
        ciborium::Value::Integer(4.into()),
    );
    params.insert("szip_rsi".to_string(), ciborium::Value::Integer(128.into()));
    params.insert(
        "szip_block_size".to_string(),
        ciborium::Value::Integer(16.into()),
    );
    params.insert(
        "szip_flags".to_string(),
        ciborium::Value::Integer(8_i64.into()),
    );

    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![1024],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "shuffle".to_string(),
        compression: "szip".to_string(),
        params,
        masks: None,
    };

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let err = decode_range(&encoded, 0, &[(0, 10)], &DecodeOptions::default()).unwrap_err();
    assert!(err.to_string().contains("shuffle") || err.to_string().contains("filter"));
}

#[test]
fn test_szip_multi_object_mixed_compression() {
    // Object 0: raw uncompressed float32
    // Object 1: simple_packing + szip float64
    let values: Vec<f64> = (0..2048).map(|i| 100.0 + i as f64 * 0.1).collect();
    let packing = simple_packing::compute_params(&values, 16, 0).unwrap();

    let raw_data = vec![0u8; 10 * 4]; // 10 float32

    let mut packing_params = BTreeMap::new();
    packing_params.insert(
        "reference_value".to_string(),
        ciborium::Value::Float(packing.reference_value),
    );
    packing_params.insert(
        "binary_scale_factor".to_string(),
        ciborium::Value::Integer((packing.binary_scale_factor as i64).into()),
    );
    packing_params.insert(
        "decimal_scale_factor".to_string(),
        ciborium::Value::Integer((packing.decimal_scale_factor as i64).into()),
    );
    packing_params.insert(
        "bits_per_value".to_string(),
        ciborium::Value::Integer((packing.bits_per_value as i64).into()),
    );
    packing_params.insert("szip_rsi".to_string(), ciborium::Value::Integer(128.into()));
    packing_params.insert(
        "szip_block_size".to_string(),
        ciborium::Value::Integer(16.into()),
    );
    packing_params.insert(
        "szip_flags".to_string(),
        ciborium::Value::Integer(8_i64.into()),
    );

    let packed_data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };
    let raw_desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![10],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    let packed_desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![2048],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::native(),
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "szip".to_string(),
        params: packing_params,
        masks: None,
    };

    let encoded = encode(
        &global,
        &[(&raw_desc, &raw_data), (&packed_desc, &packed_data)],
        &EncodeOptions::default(),
    )
    .unwrap();

    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 2);
    assert_eq!(objects[0].1, raw_data);

    let decoded_values: Vec<f64> = objects[1]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    for (orig, dec) in values.iter().zip(decoded_values.iter()) {
        assert!((orig - dec).abs() < 0.01);
    }
}

#[test]
fn test_szip_hash_verification() {
    let values: Vec<f64> = (0..2048).map(|i| i as f64).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let packing = simple_packing::compute_params(&values, 16, 0).unwrap();
    let (global, desc) = make_szip_packing_pair(2048, &packing);

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Should pass with hash verification
    let options = DecodeOptions {
        verify_hash: true,
        ..Default::default()
    };
    let (_, objects) = decode(&encoded, &options).unwrap();
    assert_eq!(objects[0].1.len(), 2048 * 8);
}

#[test]
fn test_szip_decode_range_multiple_ranges() {
    let values: Vec<f64> = (0..4096).map(|i| i as f64 * 2.5).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let packing = simple_packing::compute_params(&values, 16, 0).unwrap();
    let (global, desc) = make_szip_packing_pair(4096, &packing);

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    let full_values: Vec<f64> = objects[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();

    // Two disjoint ranges: [10..20) and [3000..3050)
    let (_, partial_parts) = decode_range(
        &encoded,
        0,
        &[(10, 10), (3000, 50)],
        &DecodeOptions::default(),
    )
    .expect("decode_range");

    // Two ranges requested → two result parts
    assert_eq!(partial_parts.len(), 2, "expected 2 parts for 2 ranges");

    // Verify split: part 0 = 10 elements * 8 bytes, part 1 = 50 elements * 8 bytes
    assert_eq!(partial_parts[0].len(), 10 * 8);
    assert_eq!(partial_parts[1].len(), 50 * 8);

    // Verify part 0 values match range [10..20)
    let part0_values: Vec<f64> = partial_parts[0]
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    for (full, partial) in full_values[10..20].iter().zip(part0_values.iter()) {
        assert!((full - partial).abs() < 1e-10);
    }

    // Verify part 1 values match range [3000..3050)
    let part1_values: Vec<f64> = partial_parts[1]
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    for (full, partial) in full_values[3000..3050].iter().zip(part1_values.iter()) {
        assert!((full - partial).abs() < 1e-10);
    }

    // Also verify the joined result has the expected total count
    let total_bytes: usize = partial_parts.iter().map(|p| p.len()).sum();
    assert_eq!(total_bytes, 60 * 8);
}

#[test]
fn test_validate_empty_obj_type() {
    let global = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "".to_string(),
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

    let data = vec![0u8; 4 * 4];
    let result = encode(&global, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_err(), "expected Err but got Ok");
}

// ── Metadata common/payload/reserved round-trip ─────────────────────────────

#[test]
fn test_metadata_base_reserved_round_trip() {
    let (_, desc) = make_float32_descriptor(vec![4]);
    let data = vec![0u8; 4 * 4];

    // Pre-populate one base entry with per-object metadata (merged common+payload).
    let mut base_entry = BTreeMap::new();
    base_entry.insert(
        "centre".to_string(),
        ciborium::Value::Text("ecmwf".to_string()),
    );
    base_entry.insert(
        "date".to_string(),
        ciborium::Value::Integer(20260404.into()),
    );
    base_entry.insert(
        "custom_key".to_string(),
        ciborium::Value::Text("hello".to_string()),
    );

    let global = GlobalMetadata {
        version: 3,
        base: vec![base_entry],
        ..Default::default()
    };

    let msg = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (decoded_meta, _) = decode(&msg, &DecodeOptions::default()).unwrap();

    assert_eq!(decoded_meta.version, 3);
    // base must have one entry with user-supplied + auto-populated keys.
    assert_eq!(decoded_meta.base.len(), 1);
    assert_eq!(
        decoded_meta.base[0].get("centre"),
        Some(&ciborium::Value::Text("ecmwf".to_string()))
    );
    assert_eq!(
        decoded_meta.base[0].get("custom_key"),
        Some(&ciborium::Value::Text("hello".to_string()))
    );
    // Encoder must auto-populate _reserved_.tensor with ndim/shape/strides/dtype.
    assert!(
        decoded_meta.base[0].contains_key("_reserved_"),
        "encoder must auto-populate _reserved_ in base entries"
    );
    // Provenance fields auto-populated by the encoder at message level.
    assert!(
        decoded_meta.reserved.contains_key("encoder"),
        "reserved must contain encoder provenance"
    );
    assert!(
        decoded_meta.reserved.contains_key("time"),
        "reserved must contain time provenance"
    );
    assert!(
        decoded_meta.reserved.contains_key("uuid"),
        "reserved must contain uuid provenance"
    );
    assert!(decoded_meta.extra.is_empty());
}

#[test]
fn test_metadata_empty_sections_not_serialized() {
    let (_, desc) = make_float32_descriptor(vec![4]);
    let data = vec![0u8; 4 * 4];

    // Only set 'extra', leave base/reserved empty
    let mut extra = BTreeMap::new();
    extra.insert(
        "mars".to_string(),
        ciborium::Value::Map(vec![(
            ciborium::Value::Text("class".to_string()),
            ciborium::Value::Text("od".to_string()),
        )]),
    );

    let global = GlobalMetadata {
        version: 3,
        extra,
        ..Default::default()
    };

    let msg = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (decoded_meta, _) = decode(&msg, &DecodeOptions::default()).unwrap();

    assert_eq!(decoded_meta.version, 3);
    // Encoder auto-populates base with one entry per object.
    assert_eq!(
        decoded_meta.base.len(),
        1,
        "encoder must create one base entry per object"
    );
    assert!(
        decoded_meta.base[0].contains_key("_reserved_"),
        "encoder must auto-populate _reserved_.tensor in base entries"
    );
    // Reserved at message level — encoder populates provenance.
    assert!(
        decoded_meta.reserved.contains_key("encoder"),
        "reserved must contain encoder provenance"
    );
    assert!(decoded_meta.extra.contains_key("mars"));
}

// ── Deep-nested metadata encode/decode round-trip ───────────────────────────

/// Helper: look up a text key in a CborValue::Map.
fn cbor_map_lookup<'a>(map: &'a ciborium::Value, key: &str) -> Option<&'a ciborium::Value> {
    if let ciborium::Value::Map(entries) = map {
        for (k, v) in entries {
            if matches!(k, ciborium::Value::Text(s) if s == key) {
                return Some(v);
            }
        }
    }
    None
}

#[test]
fn test_deep_nested_metadata_round_trip() {
    let (_, desc) = make_float32_descriptor(vec![4]);
    let data = vec![0u8; 4 * 4];

    let max_depth: usize = 20;

    // Build a nested CborValue::Map chain (21 levels total):
    //   base[0]["depth_0"] → depth_1 → depth_2 → … → depth_20 → "leaf_at_20"
    let mut value = ciborium::Value::Text(format!("leaf_at_{max_depth}"));
    for d in (0..max_depth).rev() {
        value = ciborium::Value::Map(vec![(
            ciborium::Value::Text(format!("depth_{}", d + 1)),
            value,
        )]);
    }

    // Put nested chains in a base entry (both as "depth_0" and "nested").
    let mut base_entry = BTreeMap::new();
    base_entry.insert("depth_0".to_string(), value.clone());
    base_entry.insert("nested".to_string(), value);

    let global = GlobalMetadata {
        version: 3,
        base: vec![base_entry],
        ..Default::default()
    };

    let msg = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();

    // Verify base[0] round-trips: walk depth_0.depth_1.….depth_20 == "leaf_at_20"
    let mut current = decoded.base[0]
        .get("depth_0")
        .expect("base[0] must have depth_0");
    for d in 1..=max_depth {
        let key = format!("depth_{d}");
        current = cbor_map_lookup(current, &key)
            .unwrap_or_else(|| panic!("missing key {key} at depth {d}"));
    }
    assert_eq!(
        current,
        &ciborium::Value::Text(format!("leaf_at_{max_depth}")),
        "leaf value must survive round-trip at depth {max_depth}"
    );

    // Verify the "nested" key in the same base entry round-trips the same way.
    let mut current = decoded.base[0]
        .get("nested")
        .expect("base[0] entry must have 'nested'");
    for d in 1..=max_depth {
        let key = format!("depth_{d}");
        current = cbor_map_lookup(current, &key)
            .unwrap_or_else(|| panic!("missing key {key} in nested at depth {d}"));
    }
    assert_eq!(
        current,
        &ciborium::Value::Text(format!("leaf_at_{max_depth}")),
        "nested leaf must survive round-trip at depth {max_depth}"
    );
}

// ── Phase 2: postamble 24-byte layout, seekable back-fill ───────────────────

/// Buffered encode always writes the real `total_length` into the
/// postamble's mirrored slot — confirms `assemble_message` patches
/// both slots on every message.
#[test]
fn buffered_postamble_total_length_equals_message_length() {
    use tensogram::wire::POSTAMBLE_SIZE;

    let global = GlobalMetadata {
        version: 3,
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![4],
        strides: vec![4],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    let data = vec![0u8; 16];
    let msg = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Postamble layout: [ffo u64][total_length u64][END_MAGIC 8]
    let pa_start = msg.len() - POSTAMBLE_SIZE;
    let pa_total = u64::from_be_bytes(msg[pa_start + 8..pa_start + 16].try_into().unwrap());
    assert_eq!(pa_total, msg.len() as u64, "postamble must mirror length");

    // Preamble total_length must also equal the full length.
    let pre_total = u64::from_be_bytes(msg[16..24].try_into().unwrap());
    assert_eq!(pre_total, msg.len() as u64);
    assert_eq!(pre_total, pa_total);
}

/// Streaming encoder on a non-seekable sink writes `total_length = 0`
/// in both preamble and postamble — the fallback-to-forward-scan
/// contract.  `Vec<u8>` is Write-only (not Seek), so `finish()` is
/// the only path available.
#[test]
fn streaming_non_seekable_zero_total_length_preamble_and_postamble() {
    use tensogram::streaming::StreamingEncoder;
    use tensogram::wire::POSTAMBLE_SIZE;

    let global = GlobalMetadata {
        version: 3,
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![2],
        strides: vec![4],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    let data = vec![0u8; 8];

    let buf: Vec<u8> = Vec::new();
    let mut enc = StreamingEncoder::new(buf, &global, &EncodeOptions::default()).unwrap();
    enc.write_object(&desc, &data).unwrap();
    let finished = enc.finish().unwrap();

    // Preamble total_length at offset 16..24 — must be 0 (streaming).
    let pre_total = u64::from_be_bytes(finished[16..24].try_into().unwrap());
    assert_eq!(pre_total, 0, "streaming preamble total_length must be 0");

    // Postamble total_length at offset len-16..len-8 — also 0 for
    // non-seekable sinks.
    let pa_start = finished.len() - POSTAMBLE_SIZE;
    let pa_total = u64::from_be_bytes(finished[pa_start + 8..pa_start + 16].try_into().unwrap());
    assert_eq!(
        pa_total, 0,
        "streaming postamble total_length must be 0 on non-seekable sinks"
    );

    // Decode must still work (forward scan fallback).
    let (_meta, objects) = decode(&finished, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
}

/// Streaming encoder onto a seekable sink can back-fill both
/// total_length slots via `finish_with_backfill`.  Confirms the
/// Write+Seek specialisation and the post-condition that both slots
/// equal the real message length.
#[test]
fn streaming_seekable_backfill_patches_both_total_length_slots() {
    use std::io::Cursor;
    use tensogram::streaming::StreamingEncoder;
    use tensogram::wire::POSTAMBLE_SIZE;

    let global = GlobalMetadata {
        version: 3,
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![3],
        strides: vec![4],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    let data = vec![0u8; 12];

    // Cursor<Vec<u8>> is Write + Seek.
    let cursor: Cursor<Vec<u8>> = Cursor::new(Vec::new());
    let mut enc = StreamingEncoder::new(cursor, &global, &EncodeOptions::default()).unwrap();
    enc.write_object(&desc, &data).unwrap();
    let cursor = enc.finish_with_backfill().unwrap();
    let buf = cursor.into_inner();

    // Preamble total_length back-filled.
    let pre_total = u64::from_be_bytes(buf[16..24].try_into().unwrap());
    assert_eq!(pre_total, buf.len() as u64);

    // Postamble total_length back-filled.
    let pa_start = buf.len() - POSTAMBLE_SIZE;
    let pa_total = u64::from_be_bytes(buf[pa_start + 8..pa_start + 16].try_into().unwrap());
    assert_eq!(pa_total, buf.len() as u64);
    assert_eq!(pre_total, pa_total);

    // Round-trip still works after the back-fill.
    let (_meta, objects) = decode(&buf, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
}

/// The postamble end magic is always at the last 8 bytes of the
/// message — pinning the wire contract that backward scanners rely
/// on to locate the end of a preceding message.
#[test]
fn postamble_end_magic_always_at_last_8_bytes() {
    let global = GlobalMetadata {
        version: 3,
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![1],
        strides: vec![4],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    let data = vec![0u8; 4];

    // Buffered:
    let msg = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    assert_eq!(&msg[msg.len() - 8..], b"39277777");

    // Streaming non-seekable:
    use tensogram::streaming::StreamingEncoder;
    let mut enc =
        StreamingEncoder::new(Vec::<u8>::new(), &global, &EncodeOptions::default()).unwrap();
    enc.write_object(&desc, &data).unwrap();
    let streamed = enc.finish().unwrap();
    assert_eq!(&streamed[streamed.len() - 8..], b"39277777");

    // Streaming seekable + back-fill:
    use std::io::Cursor;
    let cursor: Cursor<Vec<u8>> = Cursor::new(Vec::new());
    let mut enc = StreamingEncoder::new(cursor, &global, &EncodeOptions::default()).unwrap();
    enc.write_object(&desc, &data).unwrap();
    let backfilled = enc.finish_with_backfill().unwrap().into_inner();
    assert_eq!(&backfilled[backfilled.len() - 8..], b"39277777");
}

// ── Phase 3: bidirectional scan ─────────────────────────────────────────────

fn encode_sample(version_tag: u32) -> Vec<u8> {
    // Produce a small, well-formed message with a per-message tag so
    // callers can tell messages apart by payload.
    let global = GlobalMetadata {
        version: 3,
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![4],
        strides: vec![4],
        dtype: Dtype::Uint32,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    let data: Vec<u8> = (0..4)
        .flat_map(|i| (version_tag + i).to_le_bytes())
        .collect();
    encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap()
}

/// Concatenate 10 well-formed messages and scan them.  Bidirectional
/// scan must return the same boundaries as forward-only scan.
#[test]
fn scan_bidirectional_matches_forward_on_10_messages() {
    let mut buf = Vec::new();
    for i in 0..10 {
        buf.extend_from_slice(&encode_sample(i as u32 * 100));
    }

    let fwd_only_opts = ScanOptions {
        bidirectional: false,
        ..ScanOptions::default()
    };
    let bidir_opts = ScanOptions::default();

    let fwd_only = scan_with_options(&buf, &fwd_only_opts);
    let bidir = scan_with_options(&buf, &bidir_opts);

    assert_eq!(fwd_only.len(), 10);
    assert_eq!(bidir.len(), 10);
    assert_eq!(bidir, fwd_only, "bidir scan must preserve order");
}

/// Scan via `scan()` (default options) agrees with explicit
/// forward-only walk.
#[test]
fn scan_default_is_bidirectional_and_equivalent_to_forward() {
    let mut buf = Vec::new();
    for i in 0..5 {
        buf.extend_from_slice(&encode_sample(i));
    }
    let default = scan(&buf);
    let fwd = scan_with_options(
        &buf,
        &ScanOptions {
            bidirectional: false,
            ..ScanOptions::default()
        },
    );
    assert_eq!(default, fwd);
}

/// When a mid-file message has `postamble.total_length = 0`
/// (streaming non-seekable sink), the backward walker yields and
/// the forward walker completes the scan.  Output must still match
/// forward-only.
#[test]
fn scan_bidirectional_falls_back_on_mid_file_streaming_marker() {
    use tensogram::wire::POSTAMBLE_SIZE;

    // Build 5 well-formed messages.
    let mut msgs: Vec<Vec<u8>> = (0..5).map(encode_sample).collect();

    // Zero the postamble `total_length` of message index 2 — this
    // simulates a streaming non-seekable producer's output sitting
    // in the middle of the file.
    let mid = &mut msgs[2];
    let slot = mid.len() - 16;
    mid[slot..slot + 8].copy_from_slice(&0u64.to_be_bytes());

    let mut buf = Vec::new();
    for m in &msgs {
        buf.extend_from_slice(m);
    }

    // Bidirectional: must still return all 5 boundaries.  The
    // fact that message 2's postamble says "unknown length" doesn't
    // prevent the forward walker from completing — it uses the
    // preamble's `total_length`, which the encoder did populate
    // for the buffered-then-tampered fixture.
    let bidir = scan_with_options(&buf, &ScanOptions::default());
    let fwd = scan_with_options(
        &buf,
        &ScanOptions {
            bidirectional: false,
            ..ScanOptions::default()
        },
    );
    assert_eq!(bidir.len(), 5, "all 5 messages must be found");
    assert_eq!(bidir, fwd);

    // Sanity: the `POSTAMBLE_SIZE` constant is 24 in v3 — the
    // tamper above wrote 8 zeros at `len - 16` (the total_length
    // slot between first_footer_offset and END_MAGIC).  This is
    // just a self-check so this test breaks loudly if POSTAMBLE_SIZE
    // ever changes.
    assert_eq!(POSTAMBLE_SIZE, 24);
}

/// Scan matches under both scan directions for a 1-message buffer.
#[test]
fn scan_single_message_works_in_both_modes() {
    let buf = encode_sample(42);
    let fwd = scan_with_options(
        &buf,
        &ScanOptions {
            bidirectional: false,
            ..ScanOptions::default()
        },
    );
    let bidir = scan_with_options(&buf, &ScanOptions::default());
    assert_eq!(fwd.len(), 1);
    assert_eq!(bidir, fwd);
}

/// File-level bidirectional scan through an in-memory Cursor must
/// agree with the in-memory scan.
#[test]
fn scan_file_bidirectional_matches_in_memory() {
    use std::io::Cursor;
    use tensogram::framing;

    let mut buf = Vec::new();
    for i in 0..8 {
        buf.extend_from_slice(&encode_sample(i * 17));
    }

    let mut cursor = Cursor::new(buf.clone());
    let file_result = framing::scan_file(&mut cursor).unwrap();
    let mem_result = scan(&buf);
    assert_eq!(file_result, mem_result);
    assert_eq!(file_result.len(), 8);
}

// ── Phase 6: aggregate HashFrame policy ─────────────────────────────────────

fn encode_sample_hashed(create_header: bool, create_footer: bool) -> Vec<u8> {
    let global = GlobalMetadata {
        version: 3,
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![3],
        strides: vec![4],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    let data = vec![0u8; 12];
    let opts = EncodeOptions {
        create_header_hashes: create_header,
        create_footer_hashes: create_footer,
        ..EncodeOptions::default()
    };
    encode(&global, &[(&desc, &data)], &opts).unwrap()
}

/// Buffered encode with default options emits a `HeaderHash` frame
/// (no `FooterHash`) and the CBOR schema uses the v3 `algorithm` key.
#[test]
fn buffered_encode_default_emits_header_hash_only() {
    let msg = encode_sample_hashed(true, false);
    let decoded = tensogram::framing::decode_message(&msg).unwrap();

    // HeaderHash present, FooterHash absent via the aggregate field.
    let hf = decoded
        .hash_frame
        .as_ref()
        .expect("HeaderHash must be emitted by default");
    assert_eq!(hf.algorithm, "xxh3");
    assert_eq!(hf.hashes.len(), 1);

    // Inline slot must match the aggregate value for object 0.
    use tensogram::wire::FRAME_COMMON_FOOTER_SIZE;
    let (_desc, _payload, _masks, frame_offset) = &decoded.objects[0];
    let fh = tensogram::wire::FrameHeader::read_from(&msg[*frame_offset..]).unwrap();
    let frame_end = *frame_offset + fh.total_length as usize;
    let slot_start = frame_end - FRAME_COMMON_FOOTER_SIZE;
    let inline = u64::from_be_bytes(msg[slot_start..slot_start + 8].try_into().unwrap());
    assert_eq!(
        format!("{inline:016x}"),
        hf.hashes[0],
        "aggregate hex must equal the inline slot"
    );
}

/// `create_footer_hashes = true` places the aggregate in the
/// footer region instead.
#[test]
fn buffered_encode_footer_hashes_goes_to_footer() {
    let msg = encode_sample_hashed(false, true);
    let preamble = tensogram::wire::Preamble::read_from(&msg).unwrap();
    use tensogram::wire::MessageFlags;
    assert!(
        preamble.flags.has(MessageFlags::FOOTER_HASHES),
        "FOOTER_HASHES flag must be set"
    );
    assert!(
        !preamble.flags.has(MessageFlags::HEADER_HASHES),
        "HEADER_HASHES flag must be clear"
    );
    // The decoder surfaces the same aggregate regardless of location.
    let decoded = tensogram::framing::decode_message(&msg).unwrap();
    assert!(decoded.hash_frame.is_some());
}

/// Both flags on produce both frames; readers see the same aggregate.
#[test]
fn buffered_encode_both_flags_emits_both_aggregates() {
    let msg = encode_sample_hashed(true, true);
    let preamble = tensogram::wire::Preamble::read_from(&msg).unwrap();
    use tensogram::wire::MessageFlags;
    assert!(preamble.flags.has(MessageFlags::HEADER_HASHES));
    assert!(preamble.flags.has(MessageFlags::FOOTER_HASHES));
}

/// `hash_algorithm = None` clears HASHES_PRESENT and emits no
/// aggregate regardless of the create_* flags.
#[test]
fn buffered_encode_without_hashing_clears_aggregate() {
    let global = GlobalMetadata {
        version: 3,
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![1],
        strides: vec![4],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    let data = vec![0u8; 4];
    let opts = EncodeOptions {
        hash_algorithm: None,
        create_header_hashes: true,
        create_footer_hashes: true,
        ..EncodeOptions::default()
    };
    let msg = encode(&global, &[(&desc, &data)], &opts).unwrap();

    let preamble = tensogram::wire::Preamble::read_from(&msg).unwrap();
    use tensogram::wire::MessageFlags;
    assert!(!preamble.flags.has(MessageFlags::HASHES_PRESENT));
    assert!(!preamble.flags.has(MessageFlags::HEADER_HASHES));
    assert!(!preamble.flags.has(MessageFlags::FOOTER_HASHES));
}

// ── Phase 7: rle / roaring compression codecs on bitmask dtype ──────────────

fn bitmask_payload_128_bits() -> Vec<u8> {
    // 128 bits = 16 bytes, alternating 0xAA pattern (10101010) — gives the
    // RLE codec real runs to collapse and the roaring codec distinct
    // container candidates to choose.
    vec![0xAAu8; 16]
}

#[test]
fn compression_rle_round_trips_bitmask() {
    let global = GlobalMetadata {
        version: 3,
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![128],
        strides: vec![1],
        dtype: Dtype::Bitmask,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "rle".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    let data = bitmask_payload_128_bits();
    let msg = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (_meta, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    assert_eq!(objects[0].1, data);
    assert_eq!(objects[0].0.compression, "rle");
}

#[test]
fn compression_roaring_round_trips_bitmask() {
    let global = GlobalMetadata {
        version: 3,
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![128],
        strides: vec![1],
        dtype: Dtype::Bitmask,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "roaring".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    let data = bitmask_payload_128_bits();
    let msg = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (_meta, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    assert_eq!(objects[0].1, data);
    assert_eq!(objects[0].0.compression, "roaring");
}

#[test]
fn compression_rle_rejects_non_bitmask_dtype() {
    // Pin the dtype guard: rle on float32 is an encode-time error.
    let global = GlobalMetadata {
        version: 3,
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![4],
        strides: vec![4],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "rle".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    let data = vec![0u8; 16];
    let err = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("\"rle\"") && msg.contains("dtype=bitmask"),
        "expected rle-dtype error, got: {msg}"
    );
}

#[test]
fn compression_roaring_rejects_non_bitmask_dtype() {
    let global = GlobalMetadata {
        version: 3,
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![2],
        strides: vec![4],
        dtype: Dtype::Uint32,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "roaring".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    let data = vec![0u8; 8];
    let err = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("\"roaring\"") && msg.contains("dtype=bitmask"),
        "expected roaring-dtype error, got: {msg}"
    );
}

/// `ScanOptions.max_message_size` caps the apparent message length
/// advertised by a postamble; values beyond the cap cause the
/// backward walker to yield back to the forward walker rather
/// than jumping to a bogus offset.
#[test]
fn scan_max_message_size_caps_backward_walker() {
    // Build two small messages.  With the default cap (4 GiB) both
    // are reachable via the backward walker.  With a cap of 1 byte
    // the backward walker cannot trust any postamble's total_length
    // and must fall back to forward-only scanning.
    let mut buf = Vec::new();
    for i in 0..3 {
        buf.extend_from_slice(&encode_sample(i * 7));
    }

    let big_cap = ScanOptions {
        bidirectional: true,
        max_message_size: 4 * 1024 * 1024 * 1024,
    };
    let tiny_cap = ScanOptions {
        bidirectional: true,
        max_message_size: 1, // every message exceeds this → backward yields
    };

    let via_big = scan_with_options(&buf, &big_cap);
    let via_tiny = scan_with_options(&buf, &tiny_cap);
    let via_fwd = scan_with_options(
        &buf,
        &ScanOptions {
            bidirectional: false,
            ..ScanOptions::default()
        },
    );

    // Both bidir modes and the pure forward scanner agree on the
    // boundary list — regardless of `max_message_size`, no message
    // goes missing.
    assert_eq!(via_big.len(), 3);
    assert_eq!(via_tiny, via_fwd);
    assert_eq!(via_big, via_fwd);
}

/// Adversarial: a buffer whose last 8 bytes happen to match the
/// postamble END_MAGIC but whose preceding 16 bytes don't point
/// at a real message.  The backward walker must notice the
/// preamble-check mismatch and yield to the forward walker —
/// returning the same boundary list as pure forward scan.
#[test]
fn scan_bidirectional_shrugs_off_spurious_end_magic() {
    // Start with three well-formed messages, then append a block
    // of bytes whose tail is END_MAGIC preceded by a "postamble-
    // shaped" payload that points at a random offset inside the
    // legitimate region.
    let mut buf = Vec::new();
    for i in 0..3 {
        buf.extend_from_slice(&encode_sample(i * 11));
    }
    let good_end = buf.len();

    // Bogus trailing block: 8 B of random, then a u64 offset that
    // points into the middle of the good region (not at TENSOGRM),
    // then a u64 "total_length" that's nonsense, then END_MAGIC.
    buf.extend_from_slice(&[0xAB; 8]);
    buf.extend_from_slice(&(good_end as u64 / 3).to_be_bytes()); // bogus ffo
    buf.extend_from_slice(&(good_end as u64).to_be_bytes()); // nonsense total_length
    buf.extend_from_slice(b"39277777");

    // Bidirectional scan must still return the original three
    // messages — the backward walker detects the preamble-check
    // mismatch and falls back to forward-only.
    let bidir = scan(&buf);
    let fwd = scan_with_options(
        &buf,
        &ScanOptions {
            bidirectional: false,
            ..ScanOptions::default()
        },
    );
    assert_eq!(bidir, fwd, "bidirectional must match forward on bogus tail");
    assert_eq!(bidir.len(), 3);
}

/// A buffer smaller than `PREAMBLE_SIZE + POSTAMBLE_SIZE` (48 B)
/// cannot contain any valid message; both scan modes return empty.
#[test]
fn scan_tiny_buffer_returns_empty() {
    let tiny = vec![0u8; 47];
    assert!(scan(&tiny).is_empty());
    assert!(
        scan_with_options(
            &tiny,
            &ScanOptions {
                bidirectional: false,
                ..ScanOptions::default()
            }
        )
        .is_empty()
    );
}

/// Zero-length buffer is a trivial empty scan; must not panic on
/// the backward walker's `buf.len() - 8` slice.
#[test]
fn scan_empty_buffer_returns_empty() {
    let empty: Vec<u8> = Vec::new();
    assert!(scan(&empty).is_empty());
    assert!(
        scan_with_options(
            &empty,
            &ScanOptions {
                bidirectional: false,
                ..ScanOptions::default()
            }
        )
        .is_empty()
    );
}

/// Off-by-one on `max_message_size`: a message exactly that size
/// must be accepted; one byte over must trigger fallback.
#[test]
fn scan_max_message_size_off_by_one() {
    let msg = encode_sample(1);
    let exact_cap = ScanOptions {
        bidirectional: true,
        max_message_size: msg.len() as u64,
    };
    let one_short = ScanOptions {
        bidirectional: true,
        max_message_size: (msg.len() as u64) - 1,
    };

    let via_exact = scan_with_options(&msg, &exact_cap);
    let via_short = scan_with_options(&msg, &one_short);

    // Exact-cap: backward walker accepts and bidir returns 1 msg.
    assert_eq!(via_exact.len(), 1);
    // Short-by-one: backward walker rejects → forward-only fallback
    // still finds the message.
    assert_eq!(via_short.len(), 1);
    assert_eq!(via_exact, via_short);
}

/// Tamper only the HashFrame aggregate (not the inline slot).
/// Inline-slot verification still passes (the body is untouched),
/// but the aggregate cross-check must surface a `HashMismatch`
/// identifying the specific object whose aggregate entry lies.
#[test]
fn validate_detects_hash_frame_aggregate_tamper() {
    use tensogram::framing::{decode_message, scan};
    use tensogram::validate::{IssueCode, ValidateOptions, validate_message};
    use tensogram::wire::{FRAME_COMMON_FOOTER_SIZE, FrameHeader};

    let global = GlobalMetadata {
        version: 3,
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![4],
        strides: vec![4],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    let data = vec![0x11u8; 16];
    let mut msg = encode(
        &global,
        &[(&desc, data.as_slice())],
        &EncodeOptions {
            create_header_hashes: true,
            ..EncodeOptions::default()
        },
    )
    .unwrap();

    // Find the data-object frame's inline slot hex; then find an
    // occurrence of that exact 16-char ASCII sequence elsewhere in
    // the buffer (the HashFrame CBOR body) and flip a single hex
    // char there.  This tampers the aggregate without corrupting
    // the CBOR structure around it.
    let messages = scan(&msg);
    let (msg_off, msg_len) = messages[0];
    let msg_slice = &msg[msg_off..msg_off + msg_len];
    let decoded = decode_message(msg_slice).unwrap();
    let (_, _, _, frame_offset) = decoded.objects[0];
    let fh = FrameHeader::read_from(&msg_slice[frame_offset..]).unwrap();
    let total = fh.total_length as usize;
    let slot_start = frame_offset + total - FRAME_COMMON_FOOTER_SIZE;
    let inline = u64::from_be_bytes(msg_slice[slot_start..slot_start + 8].try_into().unwrap());
    let inline_hex = format!("{inline:016x}");

    // Locate the aggregate copy inside the HashFrame and flip
    // one character deterministically.  The search finds *all*
    // occurrences; we pick the first one that isn't the inline
    // slot (which is bytes, not ASCII — so a hex-string match in
    // the byte stream can only be the CBOR aggregate).
    let aggregate_pos = msg
        .windows(inline_hex.len())
        .position(|w| w == inline_hex.as_bytes())
        .expect("aggregate hex must appear in HashFrame CBOR");
    // Flip the first char: '0'..'e' → next digit; 'f' → '0'.
    let orig = msg[aggregate_pos];
    msg[aggregate_pos] = if orig == b'f' { b'0' } else { orig + 1 };

    // Message must still decode (inline slots untouched).
    let (_meta, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);

    // But the aggregate cross-check must fire.
    let report = validate_message(&msg, &ValidateOptions::default());
    let any_hash_mismatch = report
        .issues
        .iter()
        .any(|i| i.code == IssueCode::HashMismatch);
    assert!(
        any_hash_mismatch,
        "aggregate HashFrame tamper must trigger HashMismatch, got: {:?}",
        report.issues
    );
}
