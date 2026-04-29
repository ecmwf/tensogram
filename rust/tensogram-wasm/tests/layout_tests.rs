// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Tests for the wire-format layout helpers in
//! `rust/tensogram-wasm/src/layout.rs`.
//!
//! These exports let the TypeScript wrapper implement per-object
//! HTTP Range reads without re-parsing the wire format in TS, so the
//! tests assert both the shape of the returned JS objects and
//! byte-for-byte parity with the Rust core wire format.
//!
//! Run with: wasm-pack test --node rust/tensogram-wasm

use std::collections::BTreeMap;
use tensogram::dtype::Dtype;
use tensogram::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata};
use tensogram::wire::{
    DATA_OBJECT_FOOTER_SIZE, FRAME_HEADER_SIZE, FrameHeader, FrameType, POSTAMBLE_SIZE,
};
use tensogram_wasm::{
    decode_object_from_frame, decode_range_from_frame, parse_descriptor_cbor, parse_footer_chunk,
    parse_header_chunk, read_data_object_frame_footer, read_data_object_frame_header,
    read_postamble_info, read_preamble_info,
};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

fn descriptor(shape: Vec<u64>, dtype: Dtype) -> DataObjectDescriptor {
    let mut strides = vec![1u64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: shape.len() as u64,
        shape,
        strides,
        dtype,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    }
}

fn meta() -> GlobalMetadata {
    GlobalMetadata::default()
}

fn encode_one_object(data: Vec<u8>) -> Vec<u8> {
    let desc = descriptor(vec![(data.len() / 4) as u64], Dtype::Float32);
    tensogram::encode(
        &meta(),
        &[(&desc, &data)],
        &tensogram::EncodeOptions::default(),
    )
    .unwrap()
}

fn encode_two_objects() -> Vec<u8> {
    let data0: Vec<u8> = (0..40u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let data1: Vec<u8> = (0..20u32)
        .flat_map(|i| ((i as f32) * -0.25_f32).to_le_bytes())
        .collect();
    let d0 = descriptor(vec![40], Dtype::Float32);
    let d1 = descriptor(vec![20], Dtype::Float32);
    tensogram::encode(
        &meta(),
        &[(&d0, &data0), (&d1, &data1)],
        &tensogram::EncodeOptions::default(),
    )
    .unwrap()
}

fn get_number_field(obj: &js_sys::Object, key: &str) -> f64 {
    js_sys::Reflect::get(obj, &key.into())
        .unwrap()
        .as_f64()
        .unwrap_or_else(|| panic!("{key} not a number"))
}

fn get_bool_field(obj: &js_sys::Object, key: &str) -> bool {
    js_sys::Reflect::get(obj, &key.into())
        .unwrap()
        .as_bool()
        .unwrap_or_else(|| panic!("{key} not a boolean"))
}

fn get_bigint_as_u64(obj: &js_sys::Object, key: &str) -> u64 {
    let v = js_sys::Reflect::get(obj, &key.into()).unwrap();
    // serde_wasm_bindgen::Serializer::json_compatible() demotes u64 values
    // below MAX_SAFE_INTEGER to `number`, so accept either.
    if let Some(n) = v.as_f64() {
        return n as u64;
    }
    let bi: js_sys::BigInt = v.dyn_into().expect("expected number or bigint");
    let s = bi.to_string(10).unwrap().as_string().unwrap();
    s.parse().unwrap()
}

// ── read_preamble_info ───────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn preamble_info_exposes_flags_and_total_length() {
    let msg = encode_one_object(vec![1u8; 16]);
    let info = read_preamble_info(&msg).unwrap();
    let obj: js_sys::Object = info.dyn_into().unwrap();

    assert_eq!(get_number_field(&obj, "version") as u16, 3);
    let total = get_bigint_as_u64(&obj, "total_length");
    assert_eq!(total as usize, msg.len());
    // Default encoder: HEADER_METADATA + HEADER_INDEX + HEADER_HASHES.
    assert!(get_bool_field(&obj, "has_header_metadata"));
    assert!(get_bool_field(&obj, "has_header_index"));
    assert!(get_bool_field(&obj, "hashes_present"));
}

#[wasm_bindgen_test]
fn preamble_info_rejects_short_buffer() {
    let short = vec![0u8; 10];
    assert!(read_preamble_info(&short).is_err());
}

// ── read_postamble_info ──────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn postamble_info_end_magic_ok() {
    let msg = encode_one_object(vec![2u8; 12]);
    let tail = &msg[msg.len() - POSTAMBLE_SIZE..];
    let info = read_postamble_info(tail).unwrap();
    let obj: js_sys::Object = info.dyn_into().unwrap();
    assert!(get_bool_field(&obj, "end_magic_ok"));
    let total = get_bigint_as_u64(&obj, "total_length");
    assert_eq!(total as usize, msg.len());
    let ff = get_bigint_as_u64(&obj, "first_footer_offset");
    assert!(ff > 0 && (ff as usize) < msg.len());
}

#[wasm_bindgen_test]
fn postamble_info_flags_bad_magic() {
    let mut tail = vec![0u8; POSTAMBLE_SIZE];
    tail[16..24].copy_from_slice(b"NOPEMAGC");
    let info = read_postamble_info(&tail);
    // read_postamble_info is permissive on magic and only reports it via
    // the `end_magic_ok` flag — the Postamble::read_from itself may still
    // succeed if the structural reads are fine.  We tolerate either outcome
    // but must flag the bad magic.
    if let Ok(val) = info {
        let obj: js_sys::Object = val.dyn_into().unwrap();
        assert!(!get_bool_field(&obj, "end_magic_ok"));
    }
}

// ── parse_header_chunk / parse_footer_chunk ──────────────────────────────────

#[wasm_bindgen_test]
fn header_chunk_yields_metadata_and_index_for_default_encoder() {
    let msg = encode_two_objects();
    let parsed = parse_header_chunk(&msg).unwrap();
    let obj: js_sys::Object = parsed.dyn_into().unwrap();
    let meta_val = js_sys::Reflect::get(&obj, &"metadata".into()).unwrap();
    assert!(!meta_val.is_null(), "expected header metadata frame");
    let index_val = js_sys::Reflect::get(&obj, &"index".into()).unwrap();
    let index_obj: js_sys::Object = index_val.dyn_into().unwrap();
    let offsets_val = js_sys::Reflect::get(&index_obj, &"offsets".into()).unwrap();
    let offsets: js_sys::Array = offsets_val.dyn_into().unwrap();
    assert_eq!(offsets.length(), 2);
}

#[wasm_bindgen_test]
fn parse_footer_chunk_tolerates_truncated_tail() {
    let msg = encode_one_object(vec![5u8; 40]);
    let tail = &msg[msg.len() - POSTAMBLE_SIZE..];
    let pa_obj: js_sys::Object = read_postamble_info(tail).unwrap().dyn_into().unwrap();
    let ff = get_bigint_as_u64(&pa_obj, "first_footer_offset") as usize;
    let footer_bytes = &msg[ff..msg.len() - POSTAMBLE_SIZE];
    let truncated = &footer_bytes[..footer_bytes.len().saturating_sub(1)];
    // Truncation may yield a parse error (frame cut mid-way) or a
    // chunk with fewer frames than expected; both are acceptable.
    // What's not acceptable is a panic — this test guards against that.
    let _ = parse_footer_chunk(truncated);
}

// ── read_data_object_frame_header / _footer ──────────────────────────────────

#[wasm_bindgen_test]
fn frame_header_and_footer_round_trip() {
    let msg = encode_two_objects();
    // Find the first data-object frame by scanning for FR + type 9.
    let mut pos = 24usize;
    while pos + FRAME_HEADER_SIZE <= msg.len() {
        if &msg[pos..pos + 2] != b"FR" {
            pos += 1;
            continue;
        }
        let fh = FrameHeader::read_from(&msg[pos..]).unwrap();
        if fh.frame_type == FrameType::NTensorFrame {
            break;
        }
        let total = fh.total_length as usize;
        let aligned = (pos + total + 7) & !7;
        pos = aligned.min(msg.len());
    }
    let frame_start = pos;
    let fh = FrameHeader::read_from(&msg[frame_start..]).unwrap();
    let frame_end = frame_start + fh.total_length as usize;
    let frame_bytes = &msg[frame_start..frame_end];

    let header = read_data_object_frame_header(&frame_bytes[..FRAME_HEADER_SIZE]).unwrap();
    let h_obj: js_sys::Object = header.dyn_into().unwrap();
    assert!(get_bool_field(&h_obj, "is_data_object"));
    let total = get_bigint_as_u64(&h_obj, "total_length");
    assert_eq!(total as usize, frame_bytes.len());

    let footer_slice = &frame_bytes[frame_bytes.len() - DATA_OBJECT_FOOTER_SIZE..];
    let footer = read_data_object_frame_footer(footer_slice).unwrap();
    let f_obj: js_sys::Object = footer.dyn_into().unwrap();
    assert!(get_bool_field(&f_obj, "end_magic_ok"));
    let cbor_offset = get_bigint_as_u64(&f_obj, "cbor_offset");
    assert!((cbor_offset as usize) >= FRAME_HEADER_SIZE);
    assert!((cbor_offset as usize) < frame_bytes.len() - DATA_OBJECT_FOOTER_SIZE);
}

// ── parse_descriptor_cbor ────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn descriptor_cbor_round_trip() {
    let desc = descriptor(vec![3, 4], Dtype::Float32);
    let cbor = tensogram::metadata::object_descriptor_to_cbor(&desc).unwrap();
    let parsed = parse_descriptor_cbor(&cbor).unwrap();
    assert!(parsed.is_object());
}

// ── decode_object_from_frame ─────────────────────────────────────────────────

#[wasm_bindgen_test]
fn decode_object_from_frame_parity() {
    let data: Vec<u8> = (0..16u32)
        .flat_map(|i| ((i as f32) * 0.5_f32).to_le_bytes())
        .collect();
    let msg = encode_one_object(data.clone());

    // Extract the sole data-object frame bytes.
    let mut pos = 24usize;
    while pos + FRAME_HEADER_SIZE <= msg.len() {
        if &msg[pos..pos + 2] != b"FR" {
            pos += 1;
            continue;
        }
        let fh = FrameHeader::read_from(&msg[pos..]).unwrap();
        if fh.frame_type == FrameType::NTensorFrame {
            let end = pos + fh.total_length as usize;
            let frame_bytes = &msg[pos..end];
            let decoded = decode_object_from_frame(frame_bytes, None).unwrap();
            assert_eq!(decoded.object_count(), 1);
            let view: js_sys::Uint8Array = decoded.object_data_u8(0).unwrap();
            let got: Vec<u8> = view.to_vec();
            assert_eq!(got, data);
            return;
        }
        let aligned = (pos + fh.total_length as usize + 7) & !7;
        pos = aligned.min(msg.len());
    }
    panic!("no data-object frame found");
}

// ── decode_range_from_frame ──────────────────────────────────────────────────

#[wasm_bindgen_test]
fn decode_range_from_frame_parity() {
    let data: Vec<u8> = (0..32u32)
        .flat_map(|i| ((i as f32) * 0.125_f32).to_le_bytes())
        .collect();
    let msg = encode_one_object(data);

    let mut pos = 24usize;
    while pos + FRAME_HEADER_SIZE <= msg.len() {
        if &msg[pos..pos + 2] != b"FR" {
            pos += 1;
            continue;
        }
        let fh = FrameHeader::read_from(&msg[pos..]).unwrap();
        if fh.frame_type == FrameType::NTensorFrame {
            let end = pos + fh.total_length as usize;
            let frame_bytes = &msg[pos..end];

            let ranges = js_sys::BigUint64Array::new_with_length(4);
            ranges.set_index(0, 0);
            ranges.set_index(1, 3);
            ranges.set_index(2, 10);
            ranges.set_index(3, 4);

            let result = decode_range_from_frame(frame_bytes, &ranges).unwrap();
            let obj: js_sys::Object = result.dyn_into().unwrap();
            let parts: js_sys::Array = js_sys::Reflect::get(&obj, &"parts".into())
                .unwrap()
                .dyn_into()
                .unwrap();
            assert_eq!(parts.length(), 2);
            let part0: js_sys::Uint8Array = parts.get(0).dyn_into().unwrap();
            assert_eq!(part0.length() as usize, 3 * 4);
            let part1: js_sys::Uint8Array = parts.get(1).dyn_into().unwrap();
            assert_eq!(part1.length() as usize, 4 * 4);
            return;
        }
        let aligned = (pos + fh.total_length as usize + 7) & !7;
        pos = aligned.min(msg.len());
    }
    panic!("no data-object frame found");
}

// ── parse_header_chunk on a footer-indexed file returns nulls ────────────────

#[wasm_bindgen_test]
fn parse_footer_chunk_nulls_for_header_indexed_file() {
    // Default encoder emits metadata + index in the header, so the
    // footer region sits empty between first_footer_offset and the
    // postamble. parse_footer_chunk on that empty region must produce
    // both metadata=null and index=null without erroring.
    let msg = encode_one_object(vec![9u8; 8]);
    let tail = &msg[msg.len() - POSTAMBLE_SIZE..];
    let pa: js_sys::Object = read_postamble_info(tail).unwrap().dyn_into().unwrap();
    let ff = get_bigint_as_u64(&pa, "first_footer_offset") as usize;
    let footer_bytes = &msg[ff..msg.len() - POSTAMBLE_SIZE];
    let parsed = parse_footer_chunk(footer_bytes).unwrap();
    let obj: js_sys::Object = parsed.dyn_into().unwrap();
    let metadata_val = js_sys::Reflect::get(&obj, &"metadata".into()).unwrap();
    let index_val = js_sys::Reflect::get(&obj, &"index".into()).unwrap();
    assert!(metadata_val.is_null());
    assert!(index_val.is_null());
}
