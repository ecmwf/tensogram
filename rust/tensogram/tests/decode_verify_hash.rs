// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! 6-cell matrix tests for `DecodeOptions::verify_hash` on
//! `decode` and `decode_object`.
//!
//! The matrix (per the plan,
//! `PLAN_DECODE_HASH_VERIFICATION.md` §5.2):
//!
//! | Cell | Input                                      | Option              | Expected                                  |
//! |------|--------------------------------------------|---------------------|-------------------------------------------|
//! | A    | `hash_xxh3.tgm`                            | `verify_hash=False` | success                                   |
//! | B    | `hash_xxh3.tgm`                            | `verify_hash=True`  | success                                   |
//! | C    | `simple_f32.tgm` (no per-frame hash)       | `verify_hash=True`  | `MissingHash { object_index: 0 }`         |
//! | D    | `hash_xxh3.tgm` + flipped hash byte        | `verify_hash=True`  | `HashMismatch { object_index: Some(0) }`  |
//! | E    | `hash_xxh3.tgm` + flipped payload byte     | `verify_hash=True`  | `HashMismatch { object_index: Some(0) }`  |
//! | F    | `multi_object_xxh3.tgm` + tampered obj 1   | `verify_hash=True`  | `HashMismatch { object_index: Some(1) }`  |
//!
//! Plus negative tests asserting `decode_range` ignores the flag
//! (the `verify_hash` field's docstring is the contract; here we
//! pin the runtime behaviour).

use std::collections::BTreeMap;
use std::path::PathBuf;

use tensogram::decode::{DecodeOptions, decode, decode_object, decode_range};
use tensogram::encode::{EncodeOptions, encode};
use tensogram::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata};
use tensogram::{Dtype, TensogramError, framing, wire};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn read_golden(name: &str) -> Vec<u8> {
    std::fs::read(golden_dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn opts_no_verify() -> DecodeOptions {
    DecodeOptions {
        verify_hash: false,
        ..DecodeOptions::default()
    }
}

fn opts_verify() -> DecodeOptions {
    DecodeOptions {
        verify_hash: true,
        ..DecodeOptions::default()
    }
}

/// Build a minimal single-object message with the encoder's
/// hashing path **disabled** — every frame's `HASH_PRESENT` bit
/// is clear.  Cell C ("verify on hashless message → MissingHash")
/// fixture.
fn build_unhashed_single_object_message() -> Vec<u8> {
    let meta = GlobalMetadata::default();
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![4],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };
    let mut payload = Vec::new();
    for v in [1.0f32, 2.0, 3.0, 4.0] {
        payload.extend_from_slice(&v.to_be_bytes());
    }
    let opts = EncodeOptions {
        hashing: false,
        ..Default::default()
    };
    encode(&meta, &[(&desc, &payload)], &opts).unwrap()
}

/// Locate the byte offset within the message buffer of the i-th
/// data-object frame, plus the offset within that frame of the
/// inline hash slot.  Used by tampering tests so they don't have
/// to rebuild the message bytes from scratch.
fn locate_object_frame(buf: &[u8], object_index: usize) -> (usize, usize) {
    let messages = framing::scan(buf);
    assert_eq!(
        messages.len(),
        1,
        "this test helper assumes single-message buffers"
    );
    let (msg_offset, msg_len) = messages[0];
    let msg = &buf[msg_offset..msg_offset + msg_len];
    let mut pos = 24; // past preamble
    let mut found_objects = 0;
    while pos + wire::FRAME_HEADER_SIZE <= msg.len() {
        if &msg[pos..pos + 2] != b"FR" {
            pos += 1;
            continue;
        }
        let fh = wire::FrameHeader::read_from(&msg[pos..]).unwrap();
        if fh.frame_type.is_data_object() {
            if found_objects == object_index {
                let frame_start = msg_offset + pos;
                let frame_len = fh.total_length as usize;
                // Inline hash slot lives at `frame_end - 12`.
                let slot_offset_within_frame = frame_len - wire::FRAME_COMMON_FOOTER_SIZE;
                return (frame_start, slot_offset_within_frame);
            }
            found_objects += 1;
        }
        let aligned = (pos + fh.total_length as usize + 7) & !7;
        pos = aligned.min(msg.len());
    }
    panic!(
        "data-object frame index {object_index} not found in buffer (have {found_objects} objects)"
    );
}

/// Locate the start of the *payload region* (the bytes covered by
/// the inline hash) of the i-th data-object frame.  The payload
/// region begins immediately after the 16-byte frame header;
/// flipping a byte there is guaranteed to perturb the recomputed
/// digest.
fn locate_payload_start(buf: &[u8], object_index: usize) -> usize {
    let (frame_start, _) = locate_object_frame(buf, object_index);
    frame_start + wire::FRAME_HEADER_SIZE
}

// ── Cells A & B — `hash_xxh3.tgm`, single object, both flag values ────

#[test]
fn cell_a_decode_no_verify_succeeds_on_hashed_message() {
    let data = read_golden("hash_xxh3.tgm");
    let (_, objects) = decode(&data, &opts_no_verify()).unwrap();
    assert_eq!(objects.len(), 1);
}

#[test]
fn cell_b_decode_with_verify_succeeds_on_hashed_message() {
    let data = read_golden("hash_xxh3.tgm");
    let (_, objects) = decode(&data, &opts_verify()).unwrap();
    assert_eq!(objects.len(), 1);
}

#[test]
fn cell_b_decode_object_with_verify_succeeds_on_hashed_message() {
    let data = read_golden("hash_xxh3.tgm");
    let (_, _, _) = decode_object(&data, 0, &opts_verify()).unwrap();
}

// ── Cell C — hashless message, verify=true ────────────────────────────

#[test]
fn cell_c_decode_verify_on_unhashed_message_returns_missing_hash() {
    // Built in-test rather than read from a golden because every
    // committed `.tgm` fixture is encoded with the default
    // `hashing=true`.  See `build_unhashed_single_object_message`.
    let data = build_unhashed_single_object_message();
    let err = decode(&data, &opts_verify()).unwrap_err();
    match err {
        TensogramError::MissingHash { object_index } => {
            assert_eq!(
                object_index, 0,
                "single-object message must surface index 0"
            );
        }
        other => panic!("expected MissingHash, got: {other:?}"),
    }
}

#[test]
fn cell_c_decode_object_verify_on_unhashed_message_returns_missing_hash() {
    let data = build_unhashed_single_object_message();
    let err = decode_object(&data, 0, &opts_verify()).unwrap_err();
    match err {
        TensogramError::MissingHash { object_index } => {
            assert_eq!(object_index, 0);
        }
        other => panic!("expected MissingHash, got: {other:?}"),
    }
}

#[test]
fn cell_c_no_verify_silently_decodes_unhashed_message() {
    // Documents the asymmetry: without `verify_hash` the decoder
    // never looks at the flag, so an unhashed message is
    // perfectly decodable.
    let data = build_unhashed_single_object_message();
    let (_, objects) = decode(&data, &opts_no_verify()).unwrap();
    assert_eq!(objects.len(), 1);
}

// ── Cell D — tampered hash slot (flag set, slot wrong) ────────────────

#[test]
fn cell_d_decode_verify_reports_hash_mismatch_on_tampered_slot() {
    let mut data = read_golden("hash_xxh3.tgm");
    let (frame_start, slot_offset) = locate_object_frame(&data, 0);
    // Flip one byte of the inline slot; flag bit is unchanged.
    data[frame_start + slot_offset] ^= 0xFF;

    let err = decode(&data, &opts_verify()).unwrap_err();
    match err {
        TensogramError::HashMismatch {
            object_index,
            expected,
            actual,
        } => {
            assert_eq!(object_index, Some(0));
            assert_ne!(expected, actual);
        }
        other => panic!("expected HashMismatch, got: {other:?}"),
    }
}

#[test]
fn cell_d_decode_object_verify_reports_hash_mismatch_on_tampered_slot() {
    let mut data = read_golden("hash_xxh3.tgm");
    let (frame_start, slot_offset) = locate_object_frame(&data, 0);
    data[frame_start + slot_offset] ^= 0xFF;

    let err = decode_object(&data, 0, &opts_verify()).unwrap_err();
    match err {
        TensogramError::HashMismatch {
            object_index,
            expected,
            actual,
        } => {
            assert_eq!(object_index, Some(0));
            assert_ne!(expected, actual);
        }
        other => panic!("expected HashMismatch, got: {other:?}"),
    }
}

// ── Cell E — tampered payload byte (flag set, slot intact) ────────────

#[test]
fn cell_e_decode_verify_reports_hash_mismatch_on_tampered_payload() {
    let mut data = read_golden("hash_xxh3.tgm");
    let payload_start = locate_payload_start(&data, 0);
    // Flip one byte at the start of the hashed payload region.
    data[payload_start] ^= 0xFF;

    let err = decode(&data, &opts_verify()).unwrap_err();
    match err {
        TensogramError::HashMismatch {
            object_index,
            expected,
            actual,
        } => {
            assert_eq!(object_index, Some(0));
            assert_ne!(expected, actual);
        }
        other => panic!("expected HashMismatch, got: {other:?}"),
    }
}

// ── Cell F — multi-object: tamper object 1, expect index 1 ────────────

#[test]
fn cell_f_decode_verify_reports_correct_object_index_on_multi_object() {
    let mut data = read_golden("multi_object_xxh3.tgm");
    let payload_start = locate_payload_start(&data, 1);
    data[payload_start] ^= 0xFF;

    let err = decode(&data, &opts_verify()).unwrap_err();
    match err {
        TensogramError::HashMismatch {
            object_index,
            expected,
            actual,
        } => {
            assert_eq!(
                object_index,
                Some(1),
                "must surface the *tampered* object's index, not 0"
            );
            assert_ne!(expected, actual);
        }
        other => panic!("expected HashMismatch on object 1, got: {other:?}"),
    }
}

#[test]
fn cell_f_decode_object_verify_targets_specific_object() {
    let mut data = read_golden("multi_object_xxh3.tgm");
    let payload_start = locate_payload_start(&data, 1);
    data[payload_start] ^= 0xFF;

    // Decoding object 0 still works (its hash is intact).
    decode_object(&data, 0, &opts_verify()).unwrap();
    decode_object(&data, 2, &opts_verify()).unwrap();

    // Decoding object 1 (the tampered one) reports HashMismatch.
    let err = decode_object(&data, 1, &opts_verify()).unwrap_err();
    match err {
        TensogramError::HashMismatch { object_index, .. } => {
            assert_eq!(object_index, Some(1));
        }
        other => panic!("expected HashMismatch, got: {other:?}"),
    }
}

#[test]
fn cell_f_clearing_per_frame_flag_yields_missing_hash_with_correct_index() {
    // Cross-flag-consistency case from the plan §10:
    // tampered fixture with HASHES_PRESENT=1 in preamble but
    // HASH_PRESENT=0 on object 1 → MissingHash { object_index: 1 }.
    let mut data = read_golden("multi_object_xxh3.tgm");
    let (frame_start, _) = locate_object_frame(&data, 1);
    // Frame header `flags` u16 lives at offset +6 within the frame.
    // Clear bit 1 (HASH_PRESENT) — leave bit 0 (CBOR_AFTER_PAYLOAD) alone.
    let flags_offset = frame_start + 6;
    let mut flags = u16::from_be_bytes(data[flags_offset..flags_offset + 2].try_into().unwrap());
    flags &= !wire::FrameFlags::HASH_PRESENT;
    data[flags_offset..flags_offset + 2].copy_from_slice(&flags.to_be_bytes());

    // Other objects unaffected.
    decode_object(&data, 0, &opts_verify()).unwrap();
    decode_object(&data, 2, &opts_verify()).unwrap();

    let err = decode(&data, &opts_verify()).unwrap_err();
    match err {
        TensogramError::MissingHash { object_index } => {
            assert_eq!(object_index, 1);
        }
        other => panic!("expected MissingHash on object 1, got: {other:?}"),
    }
}

// ── Negative tests: decode_range ignores verify_hash ─────────────────

#[test]
fn decode_range_ignores_verify_hash_on_unhashed_message() {
    // `simple_f32.tgm` has no per-frame hashes.  Calling
    // decode_range with `verify_hash=true` does NOT error
    // (verify_hash is silently ignored — see the doc-comment on
    // `DecodeOptions::verify_hash`).
    let data = read_golden("simple_f32.tgm");
    let (_, parts) = decode_range(&data, 0, &[(0, 4)], &opts_verify()).unwrap();
    assert_eq!(parts.len(), 1);
}

#[test]
fn decode_range_ignores_verify_hash_on_tampered_payload() {
    // Even with HASH_PRESENT set + corrupted payload, decode_range
    // does not verify and returns successfully.  This is the
    // documented contract: integrity-conscious callers must use
    // `decode_object` with `verify_hash=true`.
    let mut data = read_golden("hash_xxh3.tgm");
    let payload_start = locate_payload_start(&data, 0);
    data[payload_start] ^= 0xFF;

    let result = decode_range(&data, 0, &[(0, 1)], &opts_verify());
    // Either succeeds with corrupted bytes, or fails for a
    // non-hash reason (e.g. compression-codec error).  In every
    // case, the failure is NOT a HashMismatch / MissingHash —
    // those are the verifications we explicitly opted out of.
    match result {
        Ok(_) => (),
        Err(TensogramError::HashMismatch { .. }) | Err(TensogramError::MissingHash { .. }) => {
            panic!(
                "decode_range must not surface verify_hash errors — verify_hash is documented as ignored"
            );
        }
        Err(_) => (),
    }
}
