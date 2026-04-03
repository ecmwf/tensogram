use std::collections::BTreeMap;
use tensogram_core::*;
use tensogram_encodings::simple_packing;

fn make_float32_descriptor(shape: Vec<u64>) -> (GlobalMetadata, DataObjectDescriptor) {
    let strides = compute_strides(&shape);
    let global = GlobalMetadata {
        version: 2,
        extra: BTreeMap::new(),
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
        hash: None,
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

    let mut extra = BTreeMap::new();
    extra.insert(
        "mars".to_string(),
        ciborium::Value::Map(
            mars_global
                .into_iter()
                .map(|(k, v)| (ciborium::Value::Text(k), v))
                .collect(),
        ),
    );

    let global = GlobalMetadata { version: 2, extra };

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
        params: {
            let mut p = BTreeMap::new();
            p.insert(
                "mars_param".to_string(),
                ciborium::Value::Text(param.to_string()),
            );
            p
        },
        hash: None,
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
    assert_eq!(decoded_meta.version, 2);
    assert_eq!(decoded_objects.len(), 1);
    assert_eq!(decoded_objects[0].1, data);
}

#[test]
fn test_multi_object_message() {
    let strides1 = compute_strides(&[4, 5]);

    let global = GlobalMetadata {
        version: 2,
        extra: BTreeMap::new(),
    };

    let desc1 = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 2,
        shape: vec![4, 5],
        strides: strides1,
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
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
        hash: None,
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

    assert_eq!(meta.version, 2);
}

#[test]
fn test_decode_single_object_by_index() {
    let global = GlobalMetadata {
        version: 2,
        extra: BTreeMap::new(),
    };

    let desc1 = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![2],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    };
    let desc2 = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![3],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
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
        version: 2,
        extra: BTreeMap::new(),
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
    let options = DecodeOptions { verify_hash: true };
    let (_, objects) = decode(&encoded, &options).unwrap();
    assert_eq!(objects[0].1, data);
}

#[test]
fn test_hash_verification_fails_on_corruption() {
    // Use a larger payload so the data object frame is well-separated from
    // the metadata frames and easy to corrupt.
    let (global, desc) = make_float32_descriptor(vec![100]);
    let data = vec![42u8; 100 * 4];

    let mut encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // The v2 message layout: preamble(24) | header frames | data-object frame | footer frames | postamble(16).
    // The DataObject frame is identified by the "FR" marker followed by frame-type byte 0x00 0x04.
    // Find the data object frame and corrupt a byte inside its payload region.
    let data_frame_marker: &[u8] = &[b'F', b'R', 0x00, 0x04];
    let frame_start = encoded
        .windows(4)
        .position(|w| w == data_frame_marker)
        .expect("DataObject frame not found in encoded message");
    // Skip the 16-byte frame header to reach the payload; flip the first payload byte.
    let payload_byte = frame_start + 16;
    encoded[payload_byte] ^= 0xFF;

    let options = DecodeOptions { verify_hash: true };
    let result = decode(&encoded, &options);
    assert!(
        result.is_err(),
        "expected hash verification failure after corruption"
    );
}

#[test]
fn test_simple_packing_round_trip() {
    let values: Vec<f64> = (0..100).map(|i| 250.0 + i as f64 * 0.1).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();

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
        version: 2,
        extra: BTreeMap::new(),
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![100],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Big,
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: packing_params,
        hash: None,
    };

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();

    // Decoded values should be f64 bytes
    let decoded_values: Vec<f64> = objects[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
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
        version: 2,
        extra: BTreeMap::new(),
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![10],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "shuffle".to_string(),
        compression: "none".to_string(),
        params,
        hash: None,
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
    let partial = decode_range(&encoded, 0, &[(3, 3)], &DecodeOptions::default()).unwrap();

    let expected: Vec<u8> = values[3..6].iter().flat_map(|v| v.to_ne_bytes()).collect();
    assert_eq!(partial, expected);
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
        version: 2,
        extra: BTreeMap::new(),
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![10],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "shuffle".to_string(),
        compression: "none".to_string(),
        params,
        hash: None,
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

    // Verify mars namespace in global metadata
    assert!(meta.extra.contains_key("mars"));
    if let ciborium::Value::Map(entries) = &meta.extra["mars"] {
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
        version: 2,
        extra: BTreeMap::new(),
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
        hash: None,
    };

    let data = vec![0u8; 64];
    let result = encode(&global, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_err(), "expected Err but got Ok");
}

#[test]
fn test_cross_endian_round_trip() {
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
        version: 2,
        extra: BTreeMap::new(),
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
        hash: None,
    };

    let encoded_be = encode(&global, &[(&be_desc, &be_data)], &EncodeOptions::default()).unwrap();
    let (_, objects_be) = decode(&encoded_be, &DecodeOptions::default()).unwrap();

    let decoded_be_values: Vec<f64> = objects_be[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(decoded_be_values.len(), 50);
    for (orig, dec) in values.iter().zip(decoded_be_values.iter()) {
        assert!(
            (orig - dec).abs() < 0.01,
            "BE round-trip: orig={orig}, dec={dec}"
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
        hash: None,
    };

    let encoded_le = encode(&global, &[(&le_desc, &le_data)], &EncodeOptions::default()).unwrap();
    let (_, objects_le) = decode(&encoded_le, &DecodeOptions::default()).unwrap();

    let decoded_le_values: Vec<f64> = objects_le[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(decoded_le_values.len(), 50);
    for (orig, dec) in values.iter().zip(decoded_le_values.iter()) {
        assert!(
            (orig - dec).abs() < 0.01,
            "LE round-trip: orig={orig}, dec={dec}"
        );
    }

    assert_ne!(
        objects_be[0].1, objects_le[0].1,
        "BE and LE decoded bytes should differ"
    );
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
        version: 2,
        extra: BTreeMap::new(),
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
        params: packing_params,
        hash: None,
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
        version: 2,
        extra: BTreeMap::new(),
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 3,
        shape: vec![4, 5],
        strides: vec![5, 1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
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
        version: 2,
        extra: BTreeMap::new(),
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
        params: packing_params,
        hash: None,
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
        version: 2,
        extra: BTreeMap::new(),
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![num_values],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Big,
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "szip".to_string(),
        params: packing_params,
        hash: None,
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
        version: 2,
        extra: BTreeMap::new(),
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![num_values],
        strides: vec![1],
        dtype,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "szip".to_string(),
        params,
        hash: None,
    };
    (global, desc)
}

#[test]
fn test_szip_simple_packing_round_trip() {
    let values: Vec<f64> = (0..4096).map(|i| 250.0 + i as f64 * 0.1).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();

    let packing = simple_packing::compute_params(&values, 16, 0).unwrap();
    let (global, desc) = make_szip_packing_pair(4096, &packing);

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Verify szip_block_offsets were stored in metadata
    let (_, objects_meta) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert!(objects_meta[0].0.params.contains_key("szip_block_offsets"));

    let decoded_values: Vec<f64> = objects_meta[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(decoded_values.len(), 4096);
    for (orig, dec) in values.iter().zip(decoded_values.iter()) {
        assert!((orig - dec).abs() < 0.01, "orig={orig}, dec={dec}");
    }
}

#[test]
fn test_szip_simple_packing_decode_range_vs_full() {
    let values: Vec<f64> = (0..4096).map(|i| 100.0 + i as f64 * 0.5).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();

    let packing = simple_packing::compute_params(&values, 16, 0).unwrap();
    let (global, desc) = make_szip_packing_pair(4096, &packing);

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Full decode
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    let full_values: Vec<f64> = objects[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
        .collect();

    // Partial decode: elements 100..600 (500 elements)
    let partial_bytes =
        decode_range(&encoded, 0, &[(100, 500)], &DecodeOptions::default()).unwrap();
    let partial_values: Vec<f64> = partial_bytes
        .chunks_exact(8)
        .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
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
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();

    let packing = simple_packing::compute_params(&values, 16, 0).unwrap();
    let (global, desc) = make_szip_packing_pair(4096, &packing);

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    let full_values: Vec<f64> = objects[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
        .collect();

    // First 10 elements
    let partial_bytes = decode_range(&encoded, 0, &[(0, 10)], &DecodeOptions::default()).unwrap();
    let partial_values: Vec<f64> = partial_bytes
        .chunks_exact(8)
        .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
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
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();

    let packing = simple_packing::compute_params(&values, 16, 0).unwrap();
    let (global, desc) = make_szip_packing_pair(4096, &packing);

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    let full_values: Vec<f64> = objects[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
        .collect();

    // Last 50 elements
    let partial_bytes =
        decode_range(&encoded, 0, &[(4046, 50)], &DecodeOptions::default()).unwrap();
    let partial_values: Vec<f64> = partial_bytes
        .chunks_exact(8)
        .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
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
    let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
    let (global, desc) = make_szip_raw_pair(4096, Dtype::Uint8);

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(objects[0].1, data);
}

#[test]
fn test_szip_shuffle_round_trip() {
    // shuffle + szip: float32 data shuffled to bytes, then szip-compressed
    let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect(); // 1024 float32s
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
        version: 2,
        extra: BTreeMap::new(),
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![1024],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "shuffle".to_string(),
        compression: "szip".to_string(),
        params,
        hash: None,
    };

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(objects[0].1, data);
}

#[test]
fn test_szip_shuffle_decode_range_rejected() {
    // shuffle + szip: decode_range should be rejected
    let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
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
        version: 2,
        extra: BTreeMap::new(),
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![1024],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "shuffle".to_string(),
        compression: "szip".to_string(),
        params,
        hash: None,
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

    let packed_data: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();

    let global = GlobalMetadata {
        version: 2,
        extra: BTreeMap::new(),
    };
    let raw_desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![10],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    };
    let packed_desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![2048],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Big,
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "szip".to_string(),
        params: packing_params,
        hash: None,
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
        .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
        .collect();
    for (orig, dec) in values.iter().zip(decoded_values.iter()) {
        assert!((orig - dec).abs() < 0.01);
    }
}

#[test]
fn test_szip_hash_verification() {
    let values: Vec<f64> = (0..2048).map(|i| i as f64).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();

    let packing = simple_packing::compute_params(&values, 16, 0).unwrap();
    let (global, desc) = make_szip_packing_pair(2048, &packing);

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Should pass with hash verification
    let options = DecodeOptions { verify_hash: true };
    let (_, objects) = decode(&encoded, &options).unwrap();
    assert_eq!(objects[0].1.len(), 2048 * 8);
}

#[test]
fn test_szip_decode_range_multiple_ranges() {
    let values: Vec<f64> = (0..4096).map(|i| i as f64 * 2.5).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();

    let packing = simple_packing::compute_params(&values, 16, 0).unwrap();
    let (global, desc) = make_szip_packing_pair(4096, &packing);

    let encoded = encode(&global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    let full_values: Vec<f64> = objects[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
        .collect();

    // Two disjoint ranges: [10..20) and [3000..3050)
    let partial_bytes = decode_range(
        &encoded,
        0,
        &[(10, 10), (3000, 50)],
        &DecodeOptions::default(),
    )
    .unwrap();
    let partial_values: Vec<f64> = partial_bytes
        .chunks_exact(8)
        .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(partial_values.len(), 60);
    // First 10 values from range [10..20)
    for (full, partial) in full_values[10..20].iter().zip(partial_values[..10].iter()) {
        assert!((full - partial).abs() < 1e-10);
    }
    // Next 50 values from range [3000..3050)
    for (full, partial) in full_values[3000..3050]
        .iter()
        .zip(partial_values[10..].iter())
    {
        assert!((full - partial).abs() < 1e-10);
    }
}

#[test]
fn test_validate_empty_obj_type() {
    let global = GlobalMetadata {
        version: 2,
        extra: BTreeMap::new(),
    };
    let desc = DataObjectDescriptor {
        obj_type: "".to_string(),
        ndim: 1,
        shape: vec![4],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    };

    let data = vec![0u8; 4 * 4];
    let result = encode(&global, &[(&desc, &data)], &EncodeOptions::default());
    assert!(result.is_err(), "expected Err but got Ok");
}
