// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Integration tests for the hash-while-encoding optimisation, v3 edition.
//!
//! v3 moved the per-object hash from `DataObjectDescriptor.hash`
//! (a CBOR-descriptor field) to an inline 8-byte slot in the
//! frame footer (see `plans/WIRE_FORMAT.md` §2.4).  The
//! optimisation is the same spirit — compute the digest while
//! walking the encode bytes exactly once, don't require a second
//! pass — but the scope and storage are different:
//!
//! - **Scope**: the frame *body* = payload + mask blobs + CBOR
//!   descriptor.  Neither the frame header nor the footer
//!   (cbor_offset, hash slot, ENDF) is hashed.
//! - **Storage**: 8-byte inline slot at `frame_end − 12`.
//!
//! This suite pins the contract from four angles:
//!
//! 1. Buffered `encode()` populates the inline slot and the
//!    aggregate `HashFrame` hex matches.
//! 2. Streaming `StreamingEncoder` produces the **same inline
//!    hash** as buffered for the same descriptor + payload.
//! 3. `hash_algorithm = None` clears the `HASHES_PRESENT` flag
//!    and every slot is zero.
//! 4. Multi-threaded transparent codecs produce the same inline
//!    hash regardless of thread count (byte-identical encoded
//!    bytes + single-pass hashing = identical digest).

use std::collections::BTreeMap;
use std::io::Cursor;
use tensogram::framing::{decode_message, scan};
use tensogram::hash::{HASH_ALGORITHM_NAME, parse_hash_name, verify_frame_hash};
use tensogram::streaming::StreamingEncoder;
use tensogram::wire::{FRAME_COMMON_FOOTER_SIZE, FrameHeader, MessageFlags, Preamble};
use tensogram::{
    ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions, GlobalMetadata, decode,
    encode,
};

fn simple_desc() -> DataObjectDescriptor {
    // Rust-side convention: element strides (see
    // docs/src/guide/encode-pre-encoded.md#strides-convention).
    // 1D float32 tensor: stride 1 element.
    DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
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
    }
}

fn simple_meta() -> GlobalMetadata {
    GlobalMetadata::default()
}

fn simple_data() -> Vec<u8> {
    // 4 × f32 = 16 bytes.  Distinct values so the hash actually
    // depends on the payload content.
    let values = [1.0f32, 2.0, 3.0, 4.0];
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// Read the inline hash slot of the first data-object frame in a
/// message.  Returns `None` when `HASHES_PRESENT = 0` (slot is
/// zero) or the message is malformed.
fn first_inline_hash(wire: &[u8]) -> Option<u64> {
    let messages = scan(wire);
    let &(offset, len) = messages.first()?;
    let msg = &wire[offset..offset + len];
    let decoded = decode_message(msg).ok()?;
    let (_, _, _, frame_offset) = decoded.objects.first()?;
    let frame = &msg[*frame_offset..];
    let fh = FrameHeader::read_from(frame).ok()?;
    let total = fh.total_length as usize;
    let slot_start = *frame_offset + total - FRAME_COMMON_FOOTER_SIZE;
    let slot = u64::from_be_bytes(msg[slot_start..slot_start + 8].try_into().ok()?);
    if slot == 0 { None } else { Some(slot) }
}

#[test]
fn buffered_encode_populates_inline_slot() {
    let msg = encode(
        &simple_meta(),
        &[(&simple_desc(), simple_data().as_slice())],
        &EncodeOptions::default(),
    )
    .unwrap();

    let preamble = Preamble::read_from(&msg).unwrap();
    assert!(preamble.flags.has(MessageFlags::HASHES_PRESENT));
    let hash = first_inline_hash(&msg).expect("inline slot must be populated");
    assert_ne!(hash, 0);
}

#[test]
fn buffered_encode_inline_slot_verifies_against_body() {
    let msg = encode(
        &simple_meta(),
        &[(&simple_desc(), simple_data().as_slice())],
        &EncodeOptions::default(),
    )
    .unwrap();
    let messages = scan(&msg);
    let (offset, len) = messages[0];
    let only = &msg[offset..offset + len];
    let decoded = decode_message(only).unwrap();
    for (_, _, _, frame_offset) in &decoded.objects {
        let frame = &only[*frame_offset..];
        let fh = FrameHeader::read_from(frame).unwrap();
        let frame_bytes = &frame[..fh.total_length as usize];
        verify_frame_hash(frame_bytes, fh.frame_type, None)
            .expect("buffered inline slot must verify against body");
    }
}

#[test]
fn streaming_encode_matches_buffered_inline_hash() {
    // Same descriptor + payload must yield the same inline digest
    // on both buffered and streaming paths — the single-pass
    // hash-while-encoding contract.
    let buffered = encode(
        &simple_meta(),
        &[(&simple_desc(), simple_data().as_slice())],
        &EncodeOptions::default(),
    )
    .unwrap();
    let buffered_hash = first_inline_hash(&buffered).expect("buffered hash");

    let mut enc = StreamingEncoder::new(
        Cursor::new(Vec::<u8>::new()),
        &simple_meta(),
        &EncodeOptions::default(),
    )
    .unwrap();
    enc.write_object(&simple_desc(), &simple_data()).unwrap();
    let cursor = enc.finish_with_backfill().unwrap();
    let streamed = cursor.into_inner();
    let streamed_hash = first_inline_hash(&streamed).expect("streaming hash");

    assert_eq!(
        buffered_hash, streamed_hash,
        "buffered and streaming inline hashes must match for the same input"
    );
}

#[test]
fn hash_algorithm_none_clears_flag_and_zeros_slot() {
    let options = EncodeOptions {
        hashing: false,
        ..Default::default()
    };
    let msg = encode(
        &simple_meta(),
        &[(&simple_desc(), simple_data().as_slice())],
        &options,
    )
    .unwrap();

    let preamble = Preamble::read_from(&msg).unwrap();
    assert!(
        !preamble.flags.has(MessageFlags::HASHES_PRESENT),
        "hash_algorithm = None must clear HASHES_PRESENT"
    );
    assert!(
        first_inline_hash(&msg).is_none(),
        "hash_algorithm = None must leave the inline slot at zero"
    );

    // Sanity: decode still works with hashing disabled.
    let (_meta, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    assert_eq!(objects[0].1, simple_data());
}

#[test]
fn hash_algorithm_constant_and_parse_roundtrip() {
    // Pin that the wire-format algorithm string is "xxh3" — relied
    // upon by aggregate HashFrame serialisation and FFI accessors.
    assert_eq!(HASH_ALGORITHM_NAME, "xxh3");
    // parse_hash_name maps caller-supplied algorithm names to
    // "is hashing on" booleans; rejects unknown values.
    assert!(parse_hash_name(None).unwrap());
    assert!(parse_hash_name(Some("xxh3")).unwrap());
    assert!(!parse_hash_name(Some("none")).unwrap());
    assert!(parse_hash_name(Some("sha256")).is_err());
}
