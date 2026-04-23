// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Parity tests for `decode::decode_object_from_frame` and
//! `decode::decode_range_from_frame`.
//!
//! Both helpers take one frame's bytes in isolation (the shape a
//! caller obtains from an HTTP Range fetch keyed by the index frame).
//! They must produce byte-identical output to the corresponding
//! full-message `decode_object` / `decode_range` calls.

use std::collections::BTreeMap;

use tensogram::decode::{self, DecodeOptions, decode_object_from_frame, decode_range_from_frame};
use tensogram::encode::{EncodeOptions, encode};
use tensogram::framing;
use tensogram::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata, IndexFrame};
use tensogram::wire::{FRAME_HEADER_SIZE, FrameHeader};
use tensogram::{Dtype, metadata};

fn meta() -> GlobalMetadata {
    GlobalMetadata {
        extra: BTreeMap::new(),
        ..Default::default()
    }
}

fn descriptor(shape: Vec<u64>, compression: &str) -> DataObjectDescriptor {
    let mut strides = vec![1u64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    DataObjectDescriptor {
        obj_type: "ntensor".into(),
        ndim: shape.len() as u64,
        shape,
        strides,
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".into(),
        filter: "none".into(),
        compression: compression.into(),
        params: BTreeMap::new(),
        masks: None,
    }
}

/// Extract one data-object frame's bytes from a full message, using
/// the message's footer index.  This is exactly what a remote caller
/// would do once it had the index frame.
fn extract_object_frame(msg: &[u8], obj_idx: usize) -> &[u8] {
    let framed = framing::decode_message(msg).expect("message parses");
    let index = find_index_frame(msg).expect("message has an index");
    let off = index.offsets[obj_idx] as usize;
    let len = index.lengths[obj_idx] as usize;
    assert_eq!(framed.objects.len(), index.offsets.len());
    &msg[off..off + len]
}

fn find_index_frame(msg: &[u8]) -> Option<IndexFrame> {
    let mut pos = 24; // skip preamble
    while pos + FRAME_HEADER_SIZE <= msg.len() {
        if &msg[pos..pos + 2] != b"FR" {
            pos += 1;
            continue;
        }
        let fh = FrameHeader::read_from(&msg[pos..]).ok()?;
        let total = fh.total_length as usize;
        use tensogram::wire::FrameType;
        if matches!(
            fh.frame_type,
            FrameType::HeaderIndex | FrameType::FooterIndex
        ) {
            let payload_end = pos + total - tensogram::wire::FRAME_COMMON_FOOTER_SIZE;
            let payload = &msg[pos + FRAME_HEADER_SIZE..payload_end];
            return metadata::cbor_to_index(payload).ok();
        }
        let aligned = (pos + total + 7) & !7;
        pos = aligned.min(msg.len());
    }
    None
}

fn build_multi_object_message(compression: &str) -> Vec<u8> {
    let metadata = meta();
    let desc0 = descriptor(vec![10, 20], compression);
    let desc1 = descriptor(vec![5, 5], compression);
    let data0: Vec<u8> = (0..10usize * 20)
        .flat_map(|i| ((i as f32) * 0.25_f32).to_ne_bytes())
        .collect();
    let data1: Vec<u8> = (0..5usize * 5)
        .flat_map(|i| ((i as f32) * -0.125_f32 + 1.0).to_ne_bytes())
        .collect();

    encode(
        &metadata,
        &[(&desc0, &data0), (&desc1, &data1)],
        &EncodeOptions::default(),
    )
    .expect("encode")
}

#[test]
fn decode_object_from_frame_parity_uncompressed() {
    let msg = build_multi_object_message("none");
    let opts = DecodeOptions::default();

    for obj_idx in 0..=1 {
        let (_m, ref_desc, ref_data) =
            decode::decode_object(&msg, obj_idx, &opts).expect("full-msg decode");
        let frame = extract_object_frame(&msg, obj_idx);
        let (desc, data) = decode_object_from_frame(frame, &opts).expect("frame decode");
        assert_eq!(desc.shape, ref_desc.shape);
        assert_eq!(desc.dtype, ref_desc.dtype);
        assert_eq!(data, ref_data, "object {obj_idx} bytes diverge");
    }
}

#[test]
fn decode_object_from_frame_parity_zstd() {
    let msg = build_multi_object_message("zstd");
    let opts = DecodeOptions::default();

    for obj_idx in 0..=1 {
        let (_m, _ref_desc, ref_data) =
            decode::decode_object(&msg, obj_idx, &opts).expect("full-msg zstd decode");
        let frame = extract_object_frame(&msg, obj_idx);
        let (_desc, data) = decode_object_from_frame(frame, &opts).expect("frame zstd decode");
        assert_eq!(data, ref_data, "object {obj_idx} zstd bytes diverge");
    }
}

#[test]
fn decode_range_from_frame_parity() {
    let msg = build_multi_object_message("none");
    let opts = DecodeOptions::default();
    // Object 0 has 200 elements, object 1 has 25; both ranges fit in both.
    let ranges = [(0u64, 5u64), (10u64, 3u64)];

    for obj_idx in 0..=1 {
        let (ref_desc, ref_parts) =
            decode::decode_range(&msg, obj_idx, &ranges, &opts).expect("full-msg range");
        let frame = extract_object_frame(&msg, obj_idx);
        let (desc, parts) = decode_range_from_frame(frame, &ranges, &opts).expect("frame range");
        assert_eq!(desc.shape, ref_desc.shape);
        assert_eq!(parts.len(), ref_parts.len());
        for (a, b) in parts.iter().zip(ref_parts.iter()) {
            assert_eq!(a, b, "range bytes diverge");
        }
    }
}

#[test]
fn decode_object_from_frame_rejects_non_object_frame() {
    let meta = meta();
    let desc = descriptor(vec![4], "none");
    let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let msg = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    // Feed in the preamble bytes, which are not a frame at all.
    let not_a_frame = &msg[..24];
    let err = decode_object_from_frame(not_a_frame, &DecodeOptions::default())
        .expect_err("should reject non-frame");
    let message = format!("{err}");
    assert!(
        message.to_lowercase().contains("frame") || message.to_lowercase().contains("magic"),
        "unexpected error: {message}"
    );
}
