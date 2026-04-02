use std::collections::BTreeMap;
use tensogram_core::wire::{BinaryHeader, TERMINATOR};
use tensogram_core::*;

fn make_simple_float32_meta(shape: Vec<u64>) -> Metadata {
    let strides: Vec<u64> = if shape.is_empty() {
        vec![]
    } else {
        let mut s = vec![1u64; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            s[i] = s[i + 1] * shape[i + 1];
        }
        s
    };
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

fn make_shuffle_meta(shape: Vec<u64>, element_size: u64) -> Metadata {
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
            filter: "shuffle".to_string(),
            compression: "none".to_string(),
            params,
            hash: None,
        }],
        extra: BTreeMap::new(),
    }
}

fn craft_blob_with_header(header: BinaryHeader, total_size: usize) -> Vec<u8> {
    let mut buf = vec![0u8; total_size];
    let mut hdr_bytes = Vec::new();
    header.write_to(&mut hdr_bytes);
    buf[..hdr_bytes.len()].copy_from_slice(&hdr_bytes);
    let cbor_pos = header.metadata_offset as usize;
    if cbor_pos < total_size {
        buf[cbor_pos] = 0xA0;
    }
    let term_pos = total_size - TERMINATOR.len();
    buf[term_pos..].copy_from_slice(TERMINATOR);
    buf
}

#[test]
fn test_adversarial_non_monotonic_offsets() {
    let total_size: usize = 300;
    let fixed_header_plus_two_offsets: u64 = 56;
    let header = BinaryHeader {
        total_length: total_size as u64,
        metadata_offset: fixed_header_plus_two_offsets,
        metadata_length: 1,
        num_objects: 2,
        object_offsets: vec![200, 100],
    };
    let buf = craft_blob_with_header(header, total_size);
    let result = decode(&buf, &DecodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for non-monotonic offsets but got Ok: {:?}",
        result
    );
}

#[test]
fn test_adversarial_overlapping_regions() {
    let total_size: usize = 300;
    let cbor_offset: usize = 56;
    let obj0_start: usize = cbor_offset + 1;
    let obj1_start: usize = obj0_start + 6;

    let header = BinaryHeader {
        total_length: total_size as u64,
        metadata_offset: cbor_offset as u64,
        metadata_length: 1,
        num_objects: 2,
        object_offsets: vec![obj0_start as u64, obj1_start as u64],
    };
    let mut buf = craft_blob_with_header(header, total_size);

    buf[obj0_start..obj0_start + 4].copy_from_slice(b"OBJS");
    buf[obj1_start..obj1_start + 4].copy_from_slice(b"OBJS");
    let obje_pos = total_size - TERMINATOR.len() - 4;
    buf[obje_pos..obje_pos + 4].copy_from_slice(b"OBJE");

    let result = decode(&buf, &DecodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for overlapping object regions but got Ok: {:?}",
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

    let metadata = Metadata {
        version: 1,
        objects: vec![ObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 1,
            shape: vec![4],
            strides: vec![1],
            dtype: Dtype::Float64,
            extra: BTreeMap::new(),
        }],
        payload: vec![PayloadDescriptor {
            byte_order: ByteOrder::Big,
            encoding: "simple_packing".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params,
            hash: None,
        }],
        extra: BTreeMap::new(),
    };

    let data = vec![0u8; 4 * 8];
    let result = encode(&metadata, &[&data], &EncodeOptions::default());
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
            encoding: "simple_packing".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params,
            hash: None,
        }],
        extra: BTreeMap::new(),
    };

    let data = vec![0u8; 10 * 4];
    let result = encode(&metadata, &[&data], &EncodeOptions::default());
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
fn test_adversarial_oversized_num_objects() {
    let total_size: usize = 300;
    let fixed_header_only: u64 = 40;
    let header = BinaryHeader {
        total_length: total_size as u64,
        metadata_offset: fixed_header_only,
        metadata_length: 1,
        num_objects: u32::MAX as u64,
        object_offsets: vec![],
    };
    let buf = craft_blob_with_header(header, total_size);
    let result = decode(&buf, &DecodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for u32::MAX num_objects but got Ok: {:?}",
        result
    );
}

#[test]
fn test_adversarial_shuffle_element_size_zero() {
    let metadata = make_shuffle_meta(vec![10], 0);
    let data = vec![0u8; 10 * 4];
    let result = encode(&metadata, &[&data], &EncodeOptions::default());
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

    let metadata = make_shuffle_meta(
        vec![num_elements as u64],
        element_size_that_doesnt_divide_40,
    );
    let data = vec![0u8; num_elements * float32_byte_width];
    let result = encode(&metadata, &[&data], &EncodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for misaligned shuffle (element_size=3 on 40-byte Float32 data) but got Ok: {:?}",
        result
    );
}

#[test]
fn test_adversarial_decode_range_with_shuffle() {
    let float32_element_size: u64 = 4;
    let metadata = make_shuffle_meta(vec![10], float32_element_size);
    let data: Vec<u8> = (0u8..40).collect();

    let encoded = encode(&metadata, &[&data], &EncodeOptions::default())
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
    let mut meta = make_simple_float32_meta(vec![4]);
    meta.objects[0].obj_type = String::new();

    let data = vec![0u8; 4 * 4];
    let result = encode(&meta, &[&data], &EncodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for empty obj_type but got Ok: {:?}",
        result
    );
}

#[test]
fn test_adversarial_ndim_mismatch() {
    let metadata = Metadata {
        version: 1,
        objects: vec![ObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 5,
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
    assert!(
        result.is_err(),
        "expected Err for ndim/shape mismatch but got Ok: {:?}",
        result
    );
}
