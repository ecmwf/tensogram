// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Tests for the bidirectional remote-scan WASM exports.
//!
//! These tests pin the JS shape (`{ kind: ..., ... }` with camelCase
//! field names and bigint discipline for u64 cursor values) that the
//! TypeScript walker dispatches on.  Drift here would surface as a
//! parse error in the TS side rather than a Rust test failure, so
//! pinning the shape at the WASM boundary is the cheap regression
//! guard.
//!
//! Run with: wasm-pack test --node rust/tensogram-wasm

use tensogram::wire::{MessageFlags, POSTAMBLE_SIZE, PREAMBLE_SIZE, Postamble, Preamble, WIRE_VERSION};
use tensogram_wasm::{
    parse_backward_postamble_outcome, parse_forward_preamble_outcome, same_message_check,
    validate_backward_preamble_outcome,
};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

fn make_postamble(total_length: u64) -> Vec<u8> {
    make_postamble_with_footer(total_length, 0)
}

fn make_postamble_with_footer(total_length: u64, first_footer_offset: u64) -> Vec<u8> {
    let pa = Postamble {
        first_footer_offset,
        total_length,
    };
    let mut buf = Vec::with_capacity(POSTAMBLE_SIZE);
    pa.write_to(&mut buf);
    buf
}

fn make_preamble(total_length: u64) -> Vec<u8> {
    let pre = Preamble {
        version: WIRE_VERSION,
        flags: MessageFlags::new(0),
        reserved: 0,
        total_length,
    };
    let mut buf = Vec::with_capacity(PREAMBLE_SIZE);
    pre.write_to(&mut buf);
    buf
}

fn kind(obj: &js_sys::Object) -> String {
    js_sys::Reflect::get(obj, &"kind".into())
        .unwrap()
        .as_string()
        .expect("kind must be a string")
}

fn reason(obj: &js_sys::Object) -> String {
    js_sys::Reflect::get(obj, &"reason".into())
        .unwrap()
        .as_string()
        .expect("reason must be a string")
}

fn u64_field(obj: &js_sys::Object, key: &str) -> u64 {
    let v = js_sys::Reflect::get(obj, &key.into()).unwrap();
    if let Some(n) = v.as_f64() {
        return n as u64;
    }
    let bi: js_sys::BigInt = v.dyn_into().expect("expected number or bigint");
    bi.to_string(10).unwrap().as_string().unwrap().parse().unwrap()
}

#[wasm_bindgen_test]
fn parse_backward_postamble_short_fetch_kind_format() {
    let outcome = parse_backward_postamble_outcome(&[0u8; 4], 0, 100).unwrap();
    let obj: js_sys::Object = outcome.dyn_into().unwrap();
    assert_eq!(kind(&obj), "Format");
    assert_eq!(reason(&obj), "short-fetch-bwd");
}

#[wasm_bindgen_test]
fn parse_backward_postamble_streaming_kind() {
    let buf = make_postamble(0);
    let outcome = parse_backward_postamble_outcome(&buf, 0, POSTAMBLE_SIZE as u64).unwrap();
    let obj: js_sys::Object = outcome.dyn_into().unwrap();
    assert_eq!(kind(&obj), "Streaming");
}

#[wasm_bindgen_test]
fn parse_backward_postamble_need_validation_kind_with_camelcase_fields() {
    let buf = make_postamble(80);
    let outcome = parse_backward_postamble_outcome(&buf, 10, 100).unwrap();
    let obj: js_sys::Object = outcome.dyn_into().unwrap();
    assert_eq!(kind(&obj), "NeedPreambleValidation");
    assert_eq!(u64_field(&obj, "msgStart"), 20);
    assert_eq!(u64_field(&obj, "length"), 80);
    assert_eq!(u64_field(&obj, "firstFooterOffset"), 0);
}

#[wasm_bindgen_test]
fn parse_backward_postamble_need_validation_propagates_first_footer_offset() {
    let buf = make_postamble_with_footer(200, 96);
    let outcome = parse_backward_postamble_outcome(&buf, 0, 200).unwrap();
    let obj: js_sys::Object = outcome.dyn_into().unwrap();
    assert_eq!(kind(&obj), "NeedPreambleValidation");
    assert_eq!(u64_field(&obj, "msgStart"), 0);
    assert_eq!(u64_field(&obj, "length"), 200);
    assert_eq!(u64_field(&obj, "firstFooterOffset"), 96);
}

#[wasm_bindgen_test]
fn validate_backward_preamble_layout_kind_with_offset_length() {
    let buf = make_preamble(100);
    let outcome = validate_backward_preamble_outcome(&buf, 50, 100).unwrap();
    let obj: js_sys::Object = outcome.dyn_into().unwrap();
    assert_eq!(kind(&obj), "Layout");
    assert_eq!(u64_field(&obj, "offset"), 50);
    assert_eq!(u64_field(&obj, "length"), 100);
}

#[wasm_bindgen_test]
fn validate_backward_preamble_format_kind_on_length_mismatch() {
    let buf = make_preamble(99);
    let outcome = validate_backward_preamble_outcome(&buf, 0, 100).unwrap();
    let obj: js_sys::Object = outcome.dyn_into().unwrap();
    assert_eq!(kind(&obj), "Format");
    assert_eq!(reason(&obj), "preamble-postamble-length-mismatch");
}

#[wasm_bindgen_test]
fn parse_forward_preamble_hit_kind_with_camelcase_fields() {
    let buf = make_preamble(100);
    let outcome = parse_forward_preamble_outcome(&buf, 0, 1024, 1024).unwrap();
    let obj: js_sys::Object = outcome.dyn_into().unwrap();
    assert_eq!(kind(&obj), "Hit");
    assert_eq!(u64_field(&obj, "offset"), 0);
    assert_eq!(u64_field(&obj, "length"), 100);
    assert_eq!(u64_field(&obj, "msgEnd"), 100);
}

#[wasm_bindgen_test]
fn parse_forward_preamble_hit_beyond_bound_kind() {
    let buf = make_preamble(100);
    let outcome = parse_forward_preamble_outcome(&buf, 0, 1024, 50).unwrap();
    let obj: js_sys::Object = outcome.dyn_into().unwrap();
    assert_eq!(kind(&obj), "HitBeyondBound");
    assert_eq!(u64_field(&obj, "offset"), 0);
    assert_eq!(u64_field(&obj, "length"), 100);
    assert_eq!(u64_field(&obj, "msgEnd"), 100);
}

#[wasm_bindgen_test]
fn parse_forward_preamble_streaming_kind_with_remaining() {
    let buf = make_preamble(0);
    let outcome = parse_forward_preamble_outcome(&buf, 0, 1024, 1024).unwrap();
    let obj: js_sys::Object = outcome.dyn_into().unwrap();
    assert_eq!(kind(&obj), "Streaming");
    assert_eq!(u64_field(&obj, "remaining"), 1024);
}

#[wasm_bindgen_test]
fn parse_forward_preamble_terminate_kind_on_bad_magic() {
    let mut buf = vec![0u8; PREAMBLE_SIZE];
    buf[..8].copy_from_slice(b"NOTMAGIC");
    let outcome = parse_forward_preamble_outcome(&buf, 0, 1024, 1024).unwrap();
    let obj: js_sys::Object = outcome.dyn_into().unwrap();
    assert_eq!(kind(&obj), "Terminate");
    assert_eq!(reason(&obj), "bad-magic-fwd");
}

#[wasm_bindgen_test]
fn same_message_check_returns_bool() {
    assert!(same_message_check(0, 100, 0, 100));
    assert!(!same_message_check(0, 100, 50, 100));
    assert!(!same_message_check(0, 100, 0, 200));
}
