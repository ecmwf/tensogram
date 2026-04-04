//! Edge-case tests targeting untested code paths.
//!
//! Covers zero-element tensors, bitmask dtype, version validation,
//! NaN/Infinity params, unknown hash algorithms, empty messages, mixed dtypes,
//! decode_range edge cases, all compressor configurations, and dtype coverage.

use std::collections::BTreeMap;
use tensogram_core::*;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_global_meta() -> GlobalMetadata {
    GlobalMetadata {
        version: 2,
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
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
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
    assert_eq!(decoded_meta.version, 2);
    assert!(objects.is_empty());
}

#[test]
fn empty_message_with_custom_metadata() {
    let mut common = BTreeMap::new();
    common.insert(
        "source".to_string(),
        ciborium::Value::Text("test".to_string()),
    );
    let meta = GlobalMetadata {
        version: 2,
        common,
        ..Default::default()
    };
    let encoded = encode(&meta, &[], &EncodeOptions::default()).unwrap();
    let (decoded_meta, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert!(objects.is_empty());
    assert_eq!(
        decoded_meta.common.get("source"),
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
        hash: None,
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

// ── 8. Unknown hash algorithm on decode ──────────────────────────────────────

#[test]
fn unknown_hash_algorithm_skips_verification() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![0u8; 16];

    // Encode with a hash
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Decode normally works
    let (_, objects) = decode(&encoded, &DecodeOptions { verify_hash: true }).unwrap();
    assert!(objects[0].0.hash.is_some());

    // Now manually craft a message with an unknown hash algorithm by
    // patching the descriptor's hash_type after encoding
    // Instead: verify directly via the hash module
    let descriptor = HashDescriptor {
        hash_type: "sha512".to_string(),
        value: "fake_hash_value".to_string(),
    };
    // Should succeed (skip verification) rather than error
    assert!(tensogram_core::verify_hash(b"any data", &descriptor).is_ok());
}

// ── 9. decode_range edge cases ───────────────────────────────────────────────

#[test]
fn decode_range_empty_ranges_returns_empty() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![10], Dtype::Float32);
    let data = vec![0u8; 40]; // 10 * 4 bytes
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let result = decode_range(&encoded, 0, &[], &DecodeOptions::default()).unwrap();
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
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("unknown compression"));
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

#[test]
fn encode_without_hash() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![0u8; 16];

    let options = EncodeOptions {
        hash_algorithm: None,
    };
    let encoded = encode(&meta, &[(&desc, &data)], &options).unwrap();
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert!(objects[0].0.hash.is_none());
}

#[test]
fn verify_hash_true_on_unhashed_message_succeeds() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![0u8; 16];

    let options = EncodeOptions {
        hash_algorithm: None,
    };
    let encoded = encode(&meta, &[(&desc, &data)], &options).unwrap();
    // verify_hash: true should silently skip if no hash present
    let result = decode(&encoded, &DecodeOptions { verify_hash: true });
    assert!(result.is_ok());
}

// ── 14. decode_metadata ──────────────────────────────────────────────────────

#[test]
fn decode_metadata_only() {
    let mut common = BTreeMap::new();
    common.insert(
        "centre".to_string(),
        ciborium::Value::Text("ecmwf".to_string()),
    );
    let meta = GlobalMetadata {
        version: 2,
        common,
        ..Default::default()
    };
    let desc = make_descriptor(vec![100], Dtype::Float64);
    let data = vec![0u8; 800];
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let decoded_meta = decode_metadata(&encoded).unwrap();
    assert_eq!(
        decoded_meta.common.get("centre"),
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

    let mut file = TensogramFile::create(&path).unwrap();
    assert_eq!(file.message_count().unwrap(), 0);
}

#[test]
fn read_message_from_empty_file_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.tgm");

    let mut file = TensogramFile::create(&path).unwrap();
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

#[test]
fn hash_mismatch_detected_on_verify() {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![42u8; 16];

    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // First decode to find where the payload is, then corrupt it
    let (_, objects) = decode(&encoded, &DecodeOptions::default()).unwrap();
    assert!(objects[0].0.hash.is_some());

    // Craft a direct hash mismatch test
    let bad_hash = HashDescriptor {
        hash_type: "xxh3".to_string(),
        value: "0000000000000000".to_string(),
    };
    let result = tensogram_core::verify_hash(&data, &bad_hash);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("hash mismatch"));
}

// ── 21. GlobalMetadata namespaces ────────────────────────────────────────────

#[test]
fn metadata_namespaces_roundtrip() {
    let mut common = BTreeMap::new();
    common.insert(
        "foo".to_string(),
        ciborium::Value::Text("common_foo".to_string()),
    );
    let mut reserved = BTreeMap::new();
    reserved.insert(
        "foo".to_string(),
        ciborium::Value::Text("reserved_foo".to_string()),
    );
    let mut payload = BTreeMap::new();
    payload.insert("bar".to_string(), ciborium::Value::Integer(42.into()));

    let meta = GlobalMetadata {
        version: 2,
        common,
        payload,
        reserved,
        extra: BTreeMap::new(),
    };

    let encoded = encode(&meta, &[], &EncodeOptions::default()).unwrap();
    let (decoded, _) = decode(&encoded, &DecodeOptions::default()).unwrap();

    // Same key "foo" at different levels should be preserved
    assert_eq!(
        decoded.common.get("foo"),
        Some(&ciborium::Value::Text("common_foo".to_string()))
    );
    assert_eq!(
        decoded.reserved.get("foo"),
        Some(&ciborium::Value::Text("reserved_foo".to_string()))
    );
    assert_eq!(
        decoded.payload.get("bar"),
        Some(&ciborium::Value::Integer(42.into()))
    );
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
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: compression.to_string(),
        params,
        hash: None,
    }
}

#[test]
fn zstd_roundtrip() {
    let mut params = BTreeMap::new();
    params.insert("zstd_level".to_string(), ciborium::Value::Integer(3.into()));
    let desc = make_compressed_descriptor(vec![100], Dtype::Float32, "zstd", params);
    let data: Vec<u8> = (0..400).map(|i| (i % 256) as u8).collect();
    let (_, objects) = encode_roundtrip(&desc, &data);
    assert_eq!(objects[0].1, data);
}

#[test]
fn zstd_default_level() {
    // When zstd_level is not specified, default to 3
    let desc = make_compressed_descriptor(vec![100], Dtype::Float32, "zstd", BTreeMap::new());
    let data: Vec<u8> = (0..400).map(|i| (i % 256) as u8).collect();
    let (_, objects) = encode_roundtrip(&desc, &data);
    assert_eq!(objects[0].1, data);
}

#[test]
fn lz4_roundtrip() {
    let desc = make_compressed_descriptor(vec![100], Dtype::Float32, "lz4", BTreeMap::new());
    let data: Vec<u8> = (0..400).map(|i| (i % 256) as u8).collect();
    let (_, objects) = encode_roundtrip(&desc, &data);
    assert_eq!(objects[0].1, data);
}

#[test]
fn blosc2_roundtrip_default_codec() {
    let desc = make_compressed_descriptor(vec![100], Dtype::Float32, "blosc2", BTreeMap::new());
    let data: Vec<u8> = (0..400).map(|i| (i % 256) as u8).collect();
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
        let data: Vec<u8> = (0..400).map(|i| (i % 256) as u8).collect();
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
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("unknown blosc2 codec"));
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
    let data: Vec<u8> = (0..800).map(|i| (i % 256) as u8).collect();
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
    let data: Vec<u8> = (0..800).map(|i| (i % 256) as u8).collect();
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
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("sz3_error_bound_mode"));
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
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("unknown sz3_error_bound_mode"));
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

    let data: Vec<u8> = (0..400).map(|i| (i % 256) as u8).collect();
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

    let data: Vec<u8> = (0..400).map(|i| (i % 256) as u8).collect();
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

    let data: Vec<u8> = (0..400).map(|i| (i % 256) as u8).collect();
    let (_, objects) = encode_roundtrip(&desc, &data);
    assert_eq!(objects[0].1, data);
}

// ── 30. decode_range on non-szip (stream compressors should error) ───────────

#[test]
fn decode_range_zstd_not_supported() {
    let mut params = BTreeMap::new();
    params.insert("zstd_level".to_string(), ciborium::Value::Integer(3.into()));
    let desc = make_compressed_descriptor(vec![100], Dtype::Float32, "zstd", params);
    let data: Vec<u8> = (0..400).map(|i| (i % 256) as u8).collect();
    let meta = make_global_meta();
    let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let result = decode_range(&encoded, 0, &[(0, 10)], &DecodeOptions::default());
    // Stream compressor should return error for range decode
    assert!(result.is_err());
}

#[test]
fn decode_range_lz4_not_supported() {
    let desc = make_compressed_descriptor(vec![100], Dtype::Float32, "lz4", BTreeMap::new());
    let data: Vec<u8> = (0..400).map(|i| (i % 256) as u8).collect();
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

    let mut regular = TensogramFile::open(&path).unwrap();
    let regular_msg = regular.read_message(0).unwrap();

    let mut mmap = TensogramFile::open_mmap(&path).unwrap();
    let mmap_msg = mmap.read_message(0).unwrap();

    assert_eq!(regular_msg, mmap_msg);
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
        tensogram_core::hash::HashAlgorithm::parse("xxh3").unwrap(),
        HashAlgorithm::Xxh3
    );
}

#[test]
fn hash_algorithm_parse_invalid() {
    let result = tensogram_core::hash::HashAlgorithm::parse("md5");
    assert!(result.is_err());
}

// ── 37. compute_hash determinism ─────────────────────────────────────────────

#[test]
fn compute_hash_deterministic() {
    let data = b"hello tensogram";
    let h1 = tensogram_core::compute_hash(data, HashAlgorithm::Xxh3);
    let h2 = tensogram_core::compute_hash(data, HashAlgorithm::Xxh3);
    assert_eq!(h1, h2);
    assert_eq!(h1.len(), 16); // 64-bit hex
}

#[test]
fn compute_hash_empty_data() {
    let h = tensogram_core::compute_hash(b"", HashAlgorithm::Xxh3);
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
    let result =
        decode_range(&encoded, 0, &[(0, 5)], &DecodeOptions { verify_hash: true }).unwrap();
    assert_eq!(result.len(), 20); // 5 * 4 bytes
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
