// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Tests for the frame-walker exports in
//! `rust/tensogram-wasm/src/frame_walk.rs`.
//!
//! These cover the WASM binding layer directly — the JS object shape it
//! emits, the payload-copy contract, and the `aggregate_hash` name
//! guard, which the TypeScript wrapper's own validation shadows and so
//! cannot reach.
//!
//! Run with: wasm-pack test --node rust/tensogram-wasm

use std::collections::BTreeMap;
use tensogram::dtype::Dtype;
use tensogram::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata};
use tensogram::wire::{FRAME_HEADER_SIZE, POSTAMBLE_SIZE, PREAMBLE_SIZE};
use tensogram_wasm::{encode, frames, message_header};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

fn descriptor(shape: Vec<u64>) -> DataObjectDescriptor {
    let mut strides = vec![1u64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: shape.len() as u64,
        shape,
        strides,
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    }
}

/// A buffered (random-access) message holding `n` four-element f32
/// objects, encoded with the supplied options.
fn message_with(n: usize, options: &tensogram::EncodeOptions) -> Vec<u8> {
    let desc = descriptor(vec![4]);
    let data: Vec<u8> = (0..4u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let pairs: Vec<(&DataObjectDescriptor, &[u8])> =
        (0..n).map(|_| (&desc, data.as_slice())).collect();
    tensogram::encode(&GlobalMetadata::default(), &pairs, options).unwrap()
}

fn message(n: usize) -> Vec<u8> {
    message_with(n, &tensogram::EncodeOptions::default())
}

fn frame_at(arr: &js_sys::Array, i: u32) -> js_sys::Object {
    arr.get(i).dyn_into().expect("frame is a JS object")
}

fn string_field(obj: &js_sys::Object, key: &str) -> String {
    js_sys::Reflect::get(obj, &key.into())
        .unwrap()
        .as_string()
        .unwrap_or_else(|| panic!("{key} not a string"))
}

fn number_field(obj: &js_sys::Object, key: &str) -> f64 {
    js_sys::Reflect::get(obj, &key.into())
        .unwrap()
        .as_f64()
        .unwrap_or_else(|| panic!("{key} not a number"))
}

fn bool_field(obj: &js_sys::Object, key: &str) -> bool {
    js_sys::Reflect::get(obj, &key.into())
        .unwrap()
        .as_bool()
        .unwrap_or_else(|| panic!("{key} not a boolean"))
}

fn payload_field(obj: &js_sys::Object) -> js_sys::Uint8Array {
    js_sys::Reflect::get(obj, &"payload".into())
        .unwrap()
        .dyn_into()
        .expect("payload is a Uint8Array")
}

fn frame_type_names(msg: &[u8]) -> Vec<String> {
    let arr = frames(msg).unwrap();
    (0..arr.length())
        .map(|i| string_field(&frame_at(&arr, i), "frame_type"))
        .collect()
}

// ── frames ───────────────────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn frames_emit_canonical_wire_names_in_order() {
    assert_eq!(
        frame_type_names(&message(2)),
        vec![
            "HeaderMetadata",
            "HeaderIndex",
            "HeaderHash",
            "NTensorFrame",
            "NTensorFrame",
        ]
    );
}

#[wasm_bindgen_test]
fn frame_objects_carry_every_documented_field() {
    let msg = message(1);
    let arr = frames(&msg).unwrap();
    for i in 0..arr.length() {
        let f = frame_at(&arr, i);
        let code = number_field(&f, "frame_type_code") as u16;
        assert!((1..=9).contains(&code), "frame_type_code in the wire range");
        assert_eq!(number_field(&f, "version") as u16, 1);
        let offset = number_field(&f, "offset") as usize;
        let length = number_field(&f, "length") as usize;
        assert!(offset >= PREAMBLE_SIZE, "the preamble is not a frame");
        assert!(offset + length <= msg.len() - POSTAMBLE_SIZE);
        // has_hash mirrors bit 1 of the raw frame flags.
        let flags = number_field(&f, "flags") as u16;
        assert_eq!(bool_field(&f, "has_hash"), flags & (1 << 1) != 0);
        // The payload excludes the frame header and the type footer.
        let footer = if code == 9 { 20 } else { 12 };
        assert_eq!(
            payload_field(&f).length() as usize,
            length - FRAME_HEADER_SIZE - footer
        );
    }
}

#[wasm_bindgen_test]
fn frame_payload_is_a_js_heap_copy_not_a_view() {
    // `wasm_bindgen` materialises the `&[u8]` argument in a temporary
    // WASM allocation that is freed when `frames` returns, so a view
    // would dangle.  Round-tripping the bytes after further WASM
    // allocations proves the copy is independent.
    let msg = message(1);
    let arr = frames(&msg).unwrap();
    let f = frame_at(&arr, 0);
    let payload = payload_field(&f);
    let snapshot = payload.to_vec();
    // Allocate hard in WASM linear memory; a view would be invalidated.
    let _churn = frames(&message(64)).unwrap();
    assert_eq!(payload.to_vec(), snapshot);
    let offset = number_field(&f, "offset") as usize;
    let start = offset + FRAME_HEADER_SIZE;
    assert_eq!(snapshot, msg[start..start + snapshot.len()].to_vec());
}

#[wasm_bindgen_test]
fn frames_reject_a_buffer_that_is_not_a_message() {
    assert!(frames(b"not a tensogram message").is_err());
    assert!(frames(&[]).is_err());
    assert!(frames(&[0u8; 64]).is_err());
}

#[wasm_bindgen_test]
fn frames_reject_a_truncated_frame_chain() {
    let msg = message(2);
    let arr = frames(&msg).unwrap();
    let last = frame_at(&arr, arr.length() - 1);
    let cut = number_field(&last, "offset") as usize + 8;
    // The preamble still parses; the chain does not.  The eager array
    // must surface the error rather than return a shorter list.
    assert!(frames(&msg[..cut]).is_err());
}

// ── message_header ───────────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn message_header_flags_match_the_frames_present() {
    let msg = message(2);
    let header: js_sys::Object = message_header(&msg).unwrap().dyn_into().unwrap();
    assert_eq!(number_field(&header, "version") as u16, 3);
    assert_eq!(number_field(&header, "total_length") as usize, msg.len());

    let names = frame_type_names(&msg);
    let has = |n: &str| names.iter().any(|t| t == n);
    assert_eq!(
        bool_field(&header, "has_header_metadata"),
        has("HeaderMetadata")
    );
    assert_eq!(
        bool_field(&header, "has_footer_metadata"),
        has("FooterMetadata")
    );
    assert_eq!(bool_field(&header, "has_header_index"), has("HeaderIndex"));
    assert_eq!(bool_field(&header, "has_footer_index"), has("FooterIndex"));
    assert_eq!(bool_field(&header, "has_header_hashes"), has("HeaderHash"));
    assert_eq!(bool_field(&header, "has_footer_hashes"), has("FooterHash"));
    assert_eq!(
        bool_field(&header, "has_preceder_metadata"),
        has("PrecederMetadata")
    );
    assert!(bool_field(&header, "has_hashes_present"));
}

#[wasm_bindgen_test]
fn message_header_rejects_a_buffer_that_is_not_a_message() {
    assert!(message_header(b"not a tensogram message").is_err());
    assert!(message_header(&[]).is_err());
}

// ── aggregate hash placement, observed through the walker ────────────────────

#[wasm_bindgen_test]
fn aggregate_hash_both_emits_a_header_and_a_footer_hash_frame() {
    let options = tensogram::EncodeOptions {
        aggregate_hash: tensogram::AggregateHashPolicy::Both,
        ..Default::default()
    };
    let names = frame_type_names(&message_with(2, &options));
    assert_eq!(names.iter().filter(|t| *t == "HeaderHash").count(), 1);
    assert_eq!(names.iter().filter(|t| *t == "FooterHash").count(), 1);
    let header_at = names.iter().position(|t| t == "HeaderHash").unwrap();
    let footer_at = names.iter().position(|t| t == "FooterHash").unwrap();
    let first_object = names.iter().position(|t| t == "NTensorFrame").unwrap();
    let last_object = names.iter().rposition(|t| t == "NTensorFrame").unwrap();
    assert!(
        header_at < first_object,
        "header aggregate precedes the body"
    );
    assert!(footer_at > last_object, "footer aggregate follows the body");
}

/// Call the JS-facing `encode` with no objects and the given
/// `aggregate_hash` name — enough input to reach the option parser.
fn encode_with_aggregate_hash(
    name: Option<&str>,
) -> Result<js_sys::Uint8Array, wasm_bindgen::JsValue> {
    encode(
        js_sys::Object::new().into(),
        js_sys::Array::new(),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        name.map(str::to_string),
    )
}

#[wasm_bindgen_test]
fn encode_accepts_every_aggregate_hash_policy_name() {
    for name in ["auto", "none", "header", "footer", "both"] {
        assert!(
            encode_with_aggregate_hash(Some(name)).is_ok(),
            "policy '{name}' must be accepted"
        );
    }
    assert!(
        encode_with_aggregate_hash(None).is_ok(),
        "an omitted policy means auto"
    );
}

#[wasm_bindgen_test]
fn encode_rejects_an_unknown_aggregate_hash_policy_name() {
    // The TypeScript wrapper validates first, so this boundary guard is
    // only reachable through the raw WASM export — keep it honest here.
    let err = encode_with_aggregate_hash(Some("sideways")).unwrap_err();
    let message = js_sys::Error::from(err).message().as_string().unwrap();
    assert!(
        message.contains("unknown aggregateHash policy 'sideways'"),
        "unexpected message: {message}"
    );
    assert!(
        message.contains("auto, none, header, footer, both"),
        "the error must list the accepted names: {message}"
    );
}
