use std::collections::BTreeMap;
use tensogram_core::*;

fn make_float32_metadata(shape: Vec<u64>) -> Metadata {
    let strides = compute_strides(&shape);

    Metadata {
        version: 1,
        objects: vec![ObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: shape.len() as u64,
            shape,
            strides,
            dtype: Dtype::Float32,
            extra: BTreeMap::new(),
        }],
        payload: vec![PayloadDescriptor {
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        }],
        extra: BTreeMap::new(),
    }
}

fn make_mars_metadata(shape: Vec<u64>, param: &str) -> Metadata {
    let strides = compute_strides(&shape);
    let mut mars = BTreeMap::new();
    mars.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
    mars.insert("type".to_string(), ciborium::Value::Text("fc".to_string()));
    mars.insert(
        "date".to_string(),
        ciborium::Value::Text("20260401".to_string()),
    );

    let mut obj_mars = BTreeMap::new();
    obj_mars.insert(
        "param".to_string(),
        ciborium::Value::Text(param.to_string()),
    );

    let mut obj_extra = BTreeMap::new();
    obj_extra.insert(
        "mars".to_string(),
        ciborium::Value::Map(
            obj_mars
                .into_iter()
                .map(|(k, v)| (ciborium::Value::Text(k), v))
                .collect(),
        ),
    );

    let mut extra = BTreeMap::new();
    extra.insert(
        "mars".to_string(),
        ciborium::Value::Map(
            mars.into_iter()
                .map(|(k, v)| (ciborium::Value::Text(k), v))
                .collect(),
        ),
    );

    Metadata {
        version: 1,
        objects: vec![ObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: shape.len() as u64,
            shape,
            strides,
            dtype: Dtype::Float32,
            extra: obj_extra,
        }],
        payload: vec![PayloadDescriptor {
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        }],
        extra,
    }
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
    let metadata = make_float32_metadata(vec![10, 20]);
    let data = vec![0u8; 10 * 20 * 4]; // 200 float32 = 800 bytes

    let encoded = encode(&metadata, &[&data], &EncodeOptions::default()).unwrap();

    // Verify magic and terminator
    assert_eq!(&encoded[0..8], b"TENSOGRM");
    assert_eq!(&encoded[encoded.len() - 8..], b"39277777");

    let (decoded_meta, decoded_objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded_meta.version, 1);
    assert_eq!(decoded_objects.len(), 1);
    assert_eq!(decoded_objects[0], data);
}

#[test]
fn test_multi_object_message() {
    let strides1 = compute_strides(&[4, 5]);

    let metadata = Metadata {
        version: 1,
        objects: vec![
            ObjectDescriptor {
                obj_type: "ntensor".to_string(),
                ndim: 2,
                shape: vec![4, 5],
                strides: strides1,
                dtype: Dtype::Float32,
                extra: BTreeMap::new(),
            },
            ObjectDescriptor {
                obj_type: "ntensor".to_string(),
                ndim: 1,
                shape: vec![3],
                strides: vec![1],
                dtype: Dtype::Float64,
                extra: BTreeMap::new(),
            },
        ],
        payload: vec![
            PayloadDescriptor {
                byte_order: ByteOrder::Big,
                encoding: "none".to_string(),
                filter: "none".to_string(),
                compression: "none".to_string(),
                params: BTreeMap::new(),
                hash: None,
            },
            PayloadDescriptor {
                byte_order: ByteOrder::Little,
                encoding: "none".to_string(),
                filter: "none".to_string(),
                compression: "none".to_string(),
                params: BTreeMap::new(),
                hash: None,
            },
        ],
        extra: BTreeMap::new(),
    };

    let data1 = vec![1u8; 4 * 5 * 4]; // 20 float32
    let data2 = vec![2u8; 3 * 8]; // 3 float64

    let encoded = encode(&metadata, &[&data1, &data2], &EncodeOptions::default()).unwrap();
    let (meta, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();

    assert_eq!(meta.objects.len(), 2);
    assert_eq!(objects.len(), 2);
    assert_eq!(objects[0], data1);
    assert_eq!(objects[1], data2);
}

#[test]
fn test_decode_metadata_only() {
    let metadata = make_mars_metadata(vec![10], "2t");
    let data = vec![0u8; 10 * 4];

    let encoded = encode(&metadata, &[&data], &EncodeOptions::default()).unwrap();
    let meta = decode_metadata(&encoded).unwrap();

    assert_eq!(meta.version, 1);
    assert_eq!(meta.objects[0].shape, vec![10]);
}

#[test]
fn test_decode_single_object_by_index() {
    let metadata = Metadata {
        version: 1,
        objects: vec![
            ObjectDescriptor {
                obj_type: "ntensor".to_string(),
                ndim: 1,
                shape: vec![2],
                strides: vec![1],
                dtype: Dtype::Float32,
                extra: BTreeMap::new(),
            },
            ObjectDescriptor {
                obj_type: "ntensor".to_string(),
                ndim: 1,
                shape: vec![3],
                strides: vec![1],
                dtype: Dtype::Float32,
                extra: BTreeMap::new(),
            },
        ],
        payload: vec![
            PayloadDescriptor {
                byte_order: ByteOrder::Big,
                encoding: "none".to_string(),
                filter: "none".to_string(),
                compression: "none".to_string(),
                params: BTreeMap::new(),
                hash: None,
            },
            PayloadDescriptor {
                byte_order: ByteOrder::Big,
                encoding: "none".to_string(),
                filter: "none".to_string(),
                compression: "none".to_string(),
                params: BTreeMap::new(),
                hash: None,
            },
        ],
        extra: BTreeMap::new(),
    };

    let data1 = vec![0xAA; 2 * 4];
    let data2 = vec![0xBB; 3 * 4];

    let encoded = encode(&metadata, &[&data1, &data2], &EncodeOptions::default()).unwrap();

    let (_, obj) = decode_object(&encoded, 1, &DecodeOptions::default()).unwrap();
    assert_eq!(obj, data2);
}

#[test]
fn test_zero_object_message() {
    let metadata = Metadata {
        version: 1,
        objects: vec![],
        payload: vec![],
        extra: BTreeMap::new(),
    };

    let encoded = encode(&metadata, &[], &EncodeOptions::default()).unwrap();
    let (meta, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(meta.objects.len(), 0);
    assert_eq!(objects.len(), 0);
}

#[test]
fn test_hash_verification_passes() {
    let metadata = make_float32_metadata(vec![4]);
    let data = vec![42u8; 4 * 4];

    let encoded = encode(&metadata, &[&data], &EncodeOptions::default()).unwrap();

    // Decode with hash verification enabled
    let options = DecodeOptions { verify_hash: true };
    let (_, objects) = decode(&encoded, &options).unwrap();
    assert_eq!(objects[0], data);
}

#[test]
fn test_hash_verification_fails_on_corruption() {
    let metadata = make_float32_metadata(vec![4]);
    let data = vec![42u8; 4 * 4];

    let mut encoded = encode(&metadata, &[&data], &EncodeOptions::default()).unwrap();

    // Corrupt one byte of payload (find the OBJS marker and corrupt after it)
    let objs_pos = encoded
        .windows(4)
        .position(|w| w == b"OBJS")
        .expect("OBJS marker not found");
    encoded[objs_pos + 5] ^= 0xFF; // flip a byte in the payload

    let options = DecodeOptions { verify_hash: true };
    let result = decode(&encoded, &options);
    assert!(result.is_err());
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

    let metadata = Metadata {
        version: 1,
        objects: vec![ObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 1,
            shape: vec![100],
            strides: vec![1],
            dtype: Dtype::Float64,
            extra: BTreeMap::new(),
        }],
        payload: vec![PayloadDescriptor {
            byte_order: ByteOrder::Big,
            encoding: "simple_packing".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: packing_params,
            hash: None,
        }],
        extra: BTreeMap::new(),
    };

    let encoded = encode(&metadata, &[&data], &EncodeOptions::default()).unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();

    // Decoded values should be f64 bytes
    let decoded_values: Vec<f64> = objects[0]
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

    let metadata = Metadata {
        version: 1,
        objects: vec![ObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 1,
            shape: vec![10],
            strides: vec![1],
            dtype: Dtype::Float32,
            extra: BTreeMap::new(),
        }],
        payload: vec![PayloadDescriptor {
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "shuffle".to_string(),
            compression: "none".to_string(),
            params,
            hash: None,
        }],
        extra: BTreeMap::new(),
    };

    let encoded = encode(&metadata, &[&data], &EncodeOptions::default()).unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(objects[0], data);
}

#[test]
fn test_scan_multi_message_buffer() {
    let meta1 = make_mars_metadata(vec![4], "2t");
    let meta2 = make_mars_metadata(vec![8], "10u");
    let data1 = vec![0u8; 4 * 4];
    let data2 = vec![0u8; 8 * 4];

    let msg1 = encode(&meta1, &[&data1], &EncodeOptions::default()).unwrap();
    let msg2 = encode(&meta2, &[&data2], &EncodeOptions::default()).unwrap();

    let mut buf = Vec::new();
    buf.extend_from_slice(&msg1);
    buf.extend_from_slice(&msg2);

    let offsets = scan(&buf);
    assert_eq!(offsets.len(), 2);

    // Decode each message from the scanned offsets
    let m1 = decode_metadata(&buf[offsets[0].0..offsets[0].0 + offsets[0].1]).unwrap();
    let m2 = decode_metadata(&buf[offsets[1].0..offsets[1].0 + offsets[1].1]).unwrap();
    assert_eq!(m1.objects[0].shape, vec![4]);
    assert_eq!(m2.objects[0].shape, vec![8]);
}

#[test]
fn test_partial_range_decode_uncompressed() {
    // 10 float32 values, decode elements 3..6
    let values: Vec<f32> = (0..10).map(|i| i as f32 * 1.5).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let metadata = make_float32_metadata(vec![10]);

    let encoded = encode(&metadata, &[&data], &EncodeOptions::default()).unwrap();

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

    let metadata = Metadata {
        version: 1,
        objects: vec![ObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 1,
            shape: vec![10],
            strides: vec![1],
            dtype: Dtype::Float32,
            extra: BTreeMap::new(),
        }],
        payload: vec![PayloadDescriptor {
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "shuffle".to_string(),
            compression: "none".to_string(),
            params,
            hash: None,
        }],
        extra: BTreeMap::new(),
    };

    let encoded = encode(&metadata, &[&data], &EncodeOptions::default()).unwrap();
    let err = decode_range(&encoded, 0, &[(3, 3)], &DecodeOptions::default()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("shuffle") || msg.contains("filter"), "{msg}");
}

#[test]
fn test_objects_payload_mismatch_rejected() {
    // Manually construct metadata with mismatched lengths
    let metadata = Metadata {
        version: 1,
        objects: vec![ObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 1,
            shape: vec![4],
            strides: vec![1],
            dtype: Dtype::Float32,
            extra: BTreeMap::new(),
        }],
        payload: vec![], // mismatch!
        extra: BTreeMap::new(),
    };

    let data = vec![0u8; 16];
    let result = encode(&metadata, &[&data], &EncodeOptions::default());
    assert!(result.is_err());
}

#[test]
fn test_file_multi_message_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi.tgm");

    let mut file = TensogramFile::create(&path).unwrap();

    // Append 3 messages with different data
    for i in 0..3u8 {
        let metadata = make_float32_metadata(vec![4]);
        let data = vec![i; 4 * 4];
        file.append(&metadata, &[&data], &EncodeOptions::default())
            .unwrap();
    }

    assert_eq!(file.message_count().unwrap(), 3);

    // Read back and verify each
    for i in 0..3u8 {
        let (_, objects) = file
            .decode_message(i as usize, &DecodeOptions::default())
            .unwrap();
        assert_eq!(objects[0], vec![i; 4 * 4]);
    }
}

#[test]
fn test_namespaced_metadata_round_trip() {
    let metadata = make_mars_metadata(vec![4], "wave_spectra");
    let data = vec![0u8; 4 * 4];

    let encoded = encode(&metadata, &[&data], &EncodeOptions::default()).unwrap();
    let meta = decode_metadata(&encoded).unwrap();

    // Verify mars namespace
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
    let metadata = Metadata {
        version: 1,
        objects: vec![ObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 2,
            shape: vec![u64::MAX, 2],
            strides: vec![2, 1],
            dtype: Dtype::Float32,
            extra: BTreeMap::new(),
        }],
        payload: vec![PayloadDescriptor {
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        }],
        extra: BTreeMap::new(),
    };

    let data = vec![0u8; 64];
    let result = encode(&metadata, &[&data], &EncodeOptions::default());
    assert!(result.is_err(), "expected Err but got Ok");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("overflow"),
        "expected 'overflow' in error, got: {msg}"
    );
}

#[test]
fn test_validate_ndim_mismatch() {
    let metadata = Metadata {
        version: 1,
        objects: vec![ObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 3,
            shape: vec![4, 5],
            strides: vec![5, 1],
            dtype: Dtype::Float32,
            extra: BTreeMap::new(),
        }],
        payload: vec![PayloadDescriptor {
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        }],
        extra: BTreeMap::new(),
    };

    let data = vec![0u8; 4 * 5 * 4];
    let result = encode(&metadata, &[&data], &EncodeOptions::default());
    assert!(result.is_err(), "expected Err but got Ok");
}

#[test]
fn test_validate_empty_obj_type() {
    let metadata = Metadata {
        version: 1,
        objects: vec![ObjectDescriptor {
            obj_type: "".to_string(),
            ndim: 1,
            shape: vec![4],
            strides: vec![1],
            dtype: Dtype::Float32,
            extra: BTreeMap::new(),
        }],
        payload: vec![PayloadDescriptor {
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        }],
        extra: BTreeMap::new(),
    };

    let data = vec![0u8; 4 * 4];
    let result = encode(&metadata, &[&data], &EncodeOptions::default());
    assert!(result.is_err(), "expected Err but got Ok");
}
