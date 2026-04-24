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
    params.insert(
        "sp_reference_value".to_string(),
        ciborium::Value::Float(0.0),
    );
    params.insert(
        "sp_binary_scale_factor".to_string(),
        ciborium::Value::Integer(below_i32_min.into()),
    );
    params.insert(
        "sp_decimal_scale_factor".to_string(),
        ciborium::Value::Integer(0.into()),
    );
    params.insert(
        "sp_bits_per_value".to_string(),
        ciborium::Value::Integer(16.into()),
    );

    let global = GlobalMetadata {
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
    };

    let data = vec![0u8; 4 * 8];
    let result = encode(&global, &[(&desc, &data)], &EncodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for out-of-range binary_scale_factor but got Ok: {:?}",
        result
    );
    let msg = result.unwrap_err().to_string();
    // Error quotes the `PackingError::InvalidParams.field` — that's the
    // Rust struct field name (internal), not the wire-format `sp_*` key.
    assert!(
        msg.contains("binary_scale_factor"),
        "expected 'binary_scale_factor' in error message, got: {msg}"
    );
}

#[test]
fn test_adversarial_non_f64_simple_packing() {
    let mut params = BTreeMap::new();
    params.insert(
        "sp_reference_value".to_string(),
        ciborium::Value::Float(0.0),
    );
    params.insert(
        "sp_binary_scale_factor".to_string(),
        ciborium::Value::Integer(0.into()),
    );
    params.insert(
        "sp_decimal_scale_factor".to_string(),
        ciborium::Value::Integer(0.into()),
    );
    params.insert(
        "sp_bits_per_value".to_string(),
        ciborium::Value::Integer(16.into()),
    );

    let global = GlobalMetadata {
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
    };

    let data = vec![0u8; 4 * 5 * 4];
    let result = encode(&global, &[(&desc, &data)], &EncodeOptions::default());
    assert!(
        result.is_err(),
        "expected Err for ndim/shape mismatch but got Ok: {:?}",
        result
    );
}

// ── Phase 2: postamble integrity (v3) ───────────────────────────────────────

/// A message whose preamble says one total_length but whose postamble
/// mirrors a different value must be rejected.  Pins the v3 §7 contract
/// that the two `total_length` slots agree whenever both are non-zero.
#[test]
fn postamble_total_length_mismatch_fails() {
    let (global, desc) = make_simple_float32_pair(vec![2, 3]);
    let payload = vec![0u8; 4 * 2 * 3];
    let mut msg = encode(&global, &[(&desc, &payload)], &EncodeOptions::default()).unwrap();

    // The postamble's total_length lives at bytes [len-16, len-8).
    // Tamper with it to be one byte shorter than the real length.
    let msg_len = msg.len() as u64;
    let fake = msg_len - 1;
    let slot_start = msg.len() - 16;
    msg[slot_start..slot_start + 8].copy_from_slice(&fake.to_be_bytes());

    let result = framing::decode_message(&msg);
    assert!(
        result.is_err(),
        "expected postamble/preamble total_length mismatch to fail decode"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("postamble total_length") && err.contains("preamble total_length"),
        "expected mismatch error, got: {err}"
    );
}

/// Postamble `total_length = 0` must remain valid — the streaming
/// non-seekable-sink case produces this and readers fall back to
/// forward scan (see v3 §9.2).
#[test]
fn postamble_zero_total_length_accepted() {
    let (global, desc) = make_simple_float32_pair(vec![2]);
    let payload = vec![0u8; 4 * 2];
    let mut msg = encode(&global, &[(&desc, &payload)], &EncodeOptions::default()).unwrap();

    // Zero the postamble total_length slot — as a non-seekable
    // streaming producer would have written it.
    let slot_start = msg.len() - 16;
    msg[slot_start..slot_start + 8].copy_from_slice(&0u64.to_be_bytes());

    // Decode must succeed — zero is the documented "unknown" signal.
    let decoded = framing::decode_message(&msg).unwrap();
    assert_eq!(decoded.objects.len(), 1);
}

/// Postamble is 24 B in v3.  Pins the wire-size invariant.
#[test]
fn postamble_is_24_bytes() {
    let (global, desc) = make_simple_float32_pair(vec![1]);
    let payload = vec![0u8; 4];
    let msg = encode(&global, &[(&desc, &payload)], &EncodeOptions::default()).unwrap();

    // Last 8 bytes are the END_MAGIC; bytes [-24 .. -16) and
    // [-16 .. -8) are the two u64 fields.  Confirm magic placement.
    assert_eq!(&msg[msg.len() - 8..], b"39277777");
    // The postamble's total_length (bytes [-16..-8)) equals the full
    // message length — buffered mode always back-fills.
    let pa_total = u64::from_be_bytes(msg[msg.len() - 16..msg.len() - 8].try_into().unwrap());
    assert_eq!(pa_total, msg.len() as u64);
}

// ── Phase 4: type 4 (obsolete v2 NTensorFrame) is reserved ─────────────────

/// Hand-constructed type-4 frame embedded in an otherwise valid
/// message must fail decode with a reserved-type error.  Pins the
/// v3 contract that type 4 is rejected at the registry lookup.
#[test]
fn frame_type_4_is_rejected() {
    // Build a syntactically-valid frame but with type=4 in the
    // header.  The frame body doesn't matter — registry rejection
    // fires at the FrameType::from_u16 stage inside
    // FrameHeader::read_from.
    use tensogram::wire::{FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC};

    let body = vec![0u8; 32];
    let total_length = (FRAME_HEADER_SIZE + body.len() + FRAME_END.len()) as u64;
    let mut frame = Vec::new();
    frame.extend_from_slice(FRAME_MAGIC);
    frame.extend_from_slice(&4u16.to_be_bytes()); // type = 4 (reserved)
    frame.extend_from_slice(&1u16.to_be_bytes()); // version
    frame.extend_from_slice(&0u16.to_be_bytes()); // flags
    frame.extend_from_slice(&total_length.to_be_bytes());
    frame.extend_from_slice(&body);
    frame.extend_from_slice(FRAME_END);

    // Try to parse the frame header — must fail with reserved-type
    // message.
    let err = tensogram::wire::FrameHeader::read_from(&frame).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("reserved frame type 4"),
        "expected reserved-type-4 error, got: {msg}"
    );
    assert!(
        msg.contains("obsolete v2"),
        "expected 'obsolete v2' in the error, got: {msg}"
    );
}
