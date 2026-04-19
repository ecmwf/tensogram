// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Integration tests for the `allow_nan` / `allow_inf` encode path
//! introduced by Commit 5 of `plans/BITMASK_FRAME.md`.
//!
//! Scope of this commit: encode-side only.  The tests verify that
//! the type-9 `NTensorMaskedFrame` is emitted with a correctly
//! populated `masks` sub-map and per-kind mask sections.  NaN / Inf
//! **reconstruction on decode** is Commit 6; tests here assert the
//! decoded payload contains the substituted zeros and inspect
//! `descriptor.masks` metadata directly.

use std::collections::BTreeMap;
use tensogram::encode::MaskMethod;
use tensogram::*;

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
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        masks: None,
        params: BTreeMap::new(),
        hash: None,
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}
fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

// ── Finite-only input: allow_* is a no-op ───────────────────────────────────

#[test]
fn allow_nan_on_finite_input_produces_no_masks() {
    // Even with allow_nan=true, finite data must not cause a mask
    // section to appear.  Descriptor.masks stays None so the frame is
    // byte-compatible with the legacy layout.
    let data = f64_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let desc = make_descriptor(vec![4], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        allow_inf: true,
        hash_algorithm: None,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();
    let (_, descriptors) = decode_descriptors(&msg).unwrap();
    assert_eq!(descriptors.len(), 1);
    assert!(
        descriptors[0].masks.is_none(),
        "finite input must not produce a masks sub-map"
    );
}

// ── NaN input without allow_nan: still rejects ──────────────────────────────

#[test]
fn default_still_rejects_nan_when_allow_inf_only() {
    // allow_inf alone does not unlock NaN.
    let data = f64_bytes(&[1.0, f64::NAN]);
    let desc = make_descriptor(vec![2], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: false,
        allow_inf: true,
        ..Default::default()
    };
    let err = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap_err();
    assert!(err.to_string().contains("NaN"));
}

#[test]
fn default_still_rejects_pos_inf_when_allow_nan_only() {
    let data = f64_bytes(&[1.0, f64::INFINITY]);
    let desc = make_descriptor(vec![2], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        allow_inf: false,
        ..Default::default()
    };
    let err = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap_err();
    assert!(err.to_string().contains("+Inf"));
}

// ── End-to-end: NaN encode produces populated mask metadata ─────────────────

#[test]
fn allow_nan_f64_produces_nan_mask_with_correct_positions() {
    let data = f64_bytes(&[1.0, f64::NAN, 2.0, f64::NAN, 5.0]);
    let desc = make_descriptor(vec![5], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        // Force roaring explicitly even for small masks to avoid the
        // small-mask auto-fallback to "none" kicking in.
        nan_mask_method: MaskMethod::Roaring,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();
    let (_, descriptors) = decode_descriptors(&msg).unwrap();
    let masks = descriptors[0]
        .masks
        .as_ref()
        .expect("masks sub-map must be present");
    let nan_md = masks.nan.as_ref().expect("nan mask must be present");
    assert_eq!(nan_md.method, "roaring");
    assert!(masks.pos_inf.is_none(), "no +Inf in input");
    assert!(masks.neg_inf.is_none(), "no -Inf in input");
    // offset == encoded_payload length (pre-mask-append).  For
    // encoding=none, encoded_payload is 5*8 = 40 bytes.
    assert_eq!(nan_md.offset, 40);
    assert!(nan_md.length > 0);
}

#[test]
fn allow_nan_default_decode_restores_canonical_nan() {
    let data = f64_bytes(&[1.0, f64::NAN, 2.0, f64::NAN, 5.0]);
    let desc = make_descriptor(vec![5], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();

    // Default decode (restore_non_finite=true) restores canonical NaN
    // at the masked positions.  Finite values round-trip exactly.
    let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
    let decoded = &objects[0].1;
    let got: Vec<f64> = decoded
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got[0], 1.0);
    assert!(got[1].is_nan(), "element 1 should restore to NaN");
    assert_eq!(got[2], 2.0);
    assert!(got[3].is_nan(), "element 3 should restore to NaN");
    assert_eq!(got[4], 5.0);
}

#[test]
fn restore_honours_non_native_byte_order() {
    // Regression: pass `native_byte_order=false` on decode.  The
    // restored NaN bits must be written in the descriptor's declared
    // byte order, not the host's native order, otherwise downstream
    // consumers reading wire-order bytes see corrupted values.
    let desc = DataObjectDescriptor {
        // Force the non-native order so the bug is observable on both
        // hosts (x86_64 little-endian and ppc64/sparc big-endian).
        byte_order: match ByteOrder::native() {
            ByteOrder::Little => ByteOrder::Big,
            ByteOrder::Big => ByteOrder::Little,
        },
        ..make_descriptor(vec![3], Dtype::Float64)
    };
    let wire_data: Vec<u8> = [1.0_f64, f64::NAN, 3.0]
        .iter()
        .flat_map(|v| match desc.byte_order {
            ByteOrder::Big => v.to_be_bytes(),
            ByteOrder::Little => v.to_le_bytes(),
        })
        .collect();
    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &wire_data)], &options).unwrap();

    // Decode with native_byte_order=false — callers get wire-order bytes.
    let decode_opts = DecodeOptions {
        native_byte_order: false,
        ..Default::default()
    };
    let (_, objects) = decode(&msg, &decode_opts).unwrap();
    // Parse the returned bytes using the descriptor's byte order;
    // position 1 must round-trip as NaN.
    let parse = |chunk: &[u8]| match desc.byte_order {
        ByteOrder::Big => f64::from_be_bytes(chunk.try_into().unwrap()),
        ByteOrder::Little => f64::from_le_bytes(chunk.try_into().unwrap()),
    };
    let bytes = &objects[0].1;
    assert_eq!(parse(&bytes[0..8]), 1.0);
    assert!(parse(&bytes[8..16]).is_nan());
    assert_eq!(parse(&bytes[16..24]), 3.0);
}

#[test]
fn allow_nan_restore_disabled_returns_substituted_zero() {
    let data = f64_bytes(&[1.0, f64::NAN, 2.0, f64::NAN, 5.0]);
    let desc = make_descriptor(vec![5], Dtype::Float64);
    let enc_options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &enc_options).unwrap();

    // restore_non_finite=false returns the 0-substituted payload
    // unchanged — the on-disk representation.
    let decode_options = DecodeOptions {
        restore_non_finite: false,
        ..Default::default()
    };
    let (_, objects) = decode(&msg, &decode_options).unwrap();
    let expected = f64_bytes(&[1.0, 0.0, 2.0, 0.0, 5.0]);
    assert_eq!(objects[0].1, expected);
}

#[test]
fn allow_inf_both_signs_produces_separate_masks() {
    let data = f64_bytes(&[
        0.0,
        f64::INFINITY,
        1.0,
        f64::NEG_INFINITY,
        2.0,
        f64::INFINITY,
    ]);
    let desc = make_descriptor(vec![6], Dtype::Float64);
    let options = EncodeOptions {
        allow_inf: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();
    let (_, descriptors) = decode_descriptors(&msg).unwrap();
    let masks = descriptors[0].masks.as_ref().unwrap();
    assert!(masks.pos_inf.is_some(), "+Inf mask must be present");
    assert!(masks.neg_inf.is_some(), "-Inf mask must be present");
    assert!(masks.nan.is_none(), "no NaN in input");

    // Canonical order: +Inf mask must come before -Inf mask in the
    // payload region (matches the encoder's documented layout).
    let pos = masks.pos_inf.as_ref().unwrap();
    let neg = masks.neg_inf.as_ref().unwrap();
    assert!(
        pos.offset < neg.offset,
        "pos_inf mask ({}) must precede neg_inf mask ({}) in region",
        pos.offset,
        neg.offset
    );
    // And both must have non-zero length.
    assert!(pos.length > 0);
    assert!(neg.length > 0);
}

#[test]
fn all_three_kinds_coexist_in_one_frame() {
    let data = f64_bytes(&[
        f64::NAN,
        1.0,
        f64::INFINITY,
        2.0,
        f64::NEG_INFINITY,
        3.0,
        f64::NAN,
    ]);
    let desc = make_descriptor(vec![7], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        allow_inf: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();
    let (_, descriptors) = decode_descriptors(&msg).unwrap();
    let masks = descriptors[0].masks.as_ref().unwrap();
    assert!(masks.nan.is_some());
    assert!(masks.pos_inf.is_some());
    assert!(masks.neg_inf.is_some());

    // Offsets must be strictly increasing in the canonical order:
    // nan < +Inf < -Inf.
    let n = masks.nan.as_ref().unwrap();
    let p = masks.pos_inf.as_ref().unwrap();
    let m = masks.neg_inf.as_ref().unwrap();
    assert!(n.offset < p.offset && p.offset < m.offset);
}

// ── Mask-method selection honoured in descriptor ───────────────────────────

#[test]
fn mask_method_rle_reflected_in_descriptor() {
    let data = f64_bytes(&[1.0, f64::NAN, 2.0, f64::NAN, 3.0, f64::NAN, 4.0, f64::NAN]);
    let desc = make_descriptor(vec![8], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        nan_mask_method: MaskMethod::Rle,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();
    let (_, descriptors) = decode_descriptors(&msg).unwrap();
    let nan_md = descriptors[0].masks.as_ref().unwrap().nan.as_ref().unwrap();
    assert_eq!(nan_md.method, "rle");
}

#[test]
fn small_mask_auto_fallback_to_none() {
    // 2-element NaN mask packs to 1 byte — way under the default
    // threshold of 128 bytes — so the encoder forces method="none"
    // regardless of the requested method.
    let data = f64_bytes(&[f64::NAN, 1.0]);
    let desc = make_descriptor(vec![2], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        nan_mask_method: MaskMethod::Roaring,
        small_mask_threshold_bytes: 128, // default
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();
    let (_, descriptors) = decode_descriptors(&msg).unwrap();
    let nan_md = descriptors[0].masks.as_ref().unwrap().nan.as_ref().unwrap();
    assert_eq!(
        nan_md.method, "none",
        "small mask must auto-fallback to 'none' even when Roaring is requested"
    );
}

#[test]
fn small_mask_threshold_zero_disables_auto_fallback() {
    // With threshold=0, a 2-element mask uses the requested method
    // (roaring) rather than falling back.
    let data = f64_bytes(&[f64::NAN, 1.0]);
    let desc = make_descriptor(vec![2], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        nan_mask_method: MaskMethod::Roaring,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();
    let (_, descriptors) = decode_descriptors(&msg).unwrap();
    let nan_md = descriptors[0].masks.as_ref().unwrap().nan.as_ref().unwrap();
    assert_eq!(nan_md.method, "roaring");
}

// ── Complex dtype priority rule ─────────────────────────────────────────────

#[test]
fn complex64_nan_restored_to_both_components() {
    // (NaN + 1i), (2 + NaN*i), (3 + 4i), (5 + 6i) →
    // elements 0 and 1 go to nan mask; 2 and 3 are finite.
    let data: Vec<u8> = [f32::NAN, 1.0, 2.0, f32::NAN, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    let desc = make_descriptor(vec![4], Dtype::Complex64);
    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();
    let (_, descriptors) = decode_descriptors(&msg).unwrap();
    assert!(descriptors[0].masks.as_ref().unwrap().nan.is_some());

    // Default decode: both real and imag components restore to NaN
    // (documented lossy behaviour — bit-exact NaN payloads are NOT
    // preserved; see plans/BITMASK_FRAME.md §7.1).
    let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    let f: Vec<f32> = objects[0]
        .1
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    // Element 0 (real, imag) = (NaN, NaN)
    assert!(f[0].is_nan());
    assert!(f[1].is_nan());
    // Element 1 (real, imag) = (NaN, NaN)
    assert!(f[2].is_nan());
    assert!(f[3].is_nan());
    // Elements 2 and 3 untouched
    assert_eq!(f[4], 3.0);
    assert_eq!(f[5], 4.0);
    assert_eq!(f[6], 5.0);
    assert_eq!(f[7], 6.0);
}

// ── Float32 end-to-end ──────────────────────────────────────────────────────

#[test]
fn allow_nan_and_allow_inf_f32_end_to_end_restores_all_kinds() {
    let data = f32_bytes(&[1.0, f32::NAN, f32::INFINITY, 0.0, f32::NEG_INFINITY]);
    let desc = make_descriptor(vec![5], Dtype::Float32);
    let options = EncodeOptions {
        allow_nan: true,
        allow_inf: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();
    let (_, descriptors) = decode_descriptors(&msg).unwrap();
    let masks = descriptors[0].masks.as_ref().unwrap();
    assert!(masks.nan.is_some() && masks.pos_inf.is_some() && masks.neg_inf.is_some());

    // Decode restores all three kinds.
    let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    let got: Vec<f32> = objects[0]
        .1
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got[0], 1.0);
    assert!(got[1].is_nan());
    assert!(got[2].is_infinite() && got[2].is_sign_positive());
    assert_eq!(got[3], 0.0);
    assert!(got[4].is_infinite() && got[4].is_sign_negative());
}

// ── StreamingEncoder parity ─────────────────────────────────────────────────

#[test]
fn streaming_allow_nan_produces_equivalent_frame() {
    let data = f64_bytes(&[1.0, f64::NAN, 2.0]);
    let desc = make_descriptor(vec![3], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };

    // Buffered encode.
    let buffered = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();
    let (_, buffered_descs) = decode_descriptors(&buffered).unwrap();

    // Streaming encode.
    let buf = Vec::new();
    let mut enc = StreamingEncoder::new(buf, &make_global_meta(), &options).unwrap();
    enc.write_object(&desc, &data).unwrap();
    let streamed = enc.finish().unwrap();
    let (_, streamed_descs) = decode_descriptors(&streamed).unwrap();

    // Both paths must produce identical mask metadata.
    assert_eq!(buffered_descs[0].masks, streamed_descs[0].masks);

    // Both decode paths restore NaN at the masked position.
    let decode_opts = DecodeOptions::default();
    let (_, buffered_objects) = decode(&buffered, &decode_opts).unwrap();
    let (_, streamed_objects) = decode(&streamed, &decode_opts).unwrap();
    let check = |bytes: &[u8]| {
        let got: Vec<f64> = bytes
            .chunks_exact(8)
            .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got[0], 1.0);
        assert!(got[1].is_nan());
        assert_eq!(got[2], 2.0);
    };
    check(&buffered_objects[0].1);
    check(&streamed_objects[0].1);
}

// ── Hash verification still works with masks present ───────────────────────

#[test]
fn hash_verifies_against_substituted_payload() {
    // The hash covers the substituted encoded payload (pre-mask-append).
    // Decode with verify_hash=true must pass because the decoder
    // strips mask bytes from the payload slice before hashing.
    let data = f64_bytes(&[1.0, f64::NAN, 2.0, f64::INFINITY]);
    let desc = make_descriptor(vec![4], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        allow_inf: true,
        hash_algorithm: Some(HashAlgorithm::Xxh3),
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();

    let decode_opts = DecodeOptions {
        verify_hash: true,
        ..Default::default()
    };
    let (_, objects) =
        decode(&msg, &decode_opts).expect("hash must verify against substituted payload");
    assert_eq!(objects.len(), 1);
}

// ── Edge cases ──────────────────────────────────────────────────────────────

#[test]
fn zero_element_tensor_with_allow_nan_has_no_masks() {
    // shape=[0] means no elements to classify.  The encoder must
    // not emit a `masks` sub-map (there's nothing to mask).
    let data: Vec<u8> = vec![];
    let mut desc = make_descriptor(vec![0], Dtype::Float64);
    // shape=[0] → strides=[1] per the helper; the encoder accepts it
    // because shape-product * elem_size = 0 = data_len.
    desc.strides = vec![1];
    let options = EncodeOptions {
        allow_nan: true,
        allow_inf: true,
        hash_algorithm: None,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();
    let (_, descriptors) = decode_descriptors(&msg).unwrap();
    assert!(
        descriptors[0].masks.is_none(),
        "zero-element tensor must not emit a mask sub-map"
    );
}

#[test]
fn all_nan_payload_round_trips() {
    // Pathological but valid: every element is NaN.  The mask is
    // all-ones, which is an efficient case for every codec
    // (roaring one run-container, rle one long run, none one byte
    // per 8 bits).
    let data = f64_bytes(&[f64::NAN; 16]);
    let desc = make_descriptor(vec![16], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();
    let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    let got: Vec<f64> = objects[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert!(
        got.iter().all(|v| v.is_nan()),
        "all 16 elements must be NaN"
    );
}

#[test]
fn unknown_mask_method_at_decode_errors_clearly() {
    // If a producer claims method="bogus" in a descriptor we must
    // not silently misdecode — the decoder surfaces the method name
    // exactly once in the error.
    let data = f64_bytes(&[1.0, f64::NAN]);
    let desc = make_descriptor(vec![2], Dtype::Float64);
    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let mut msg = encode(&make_global_meta(), &[(&desc, &data)], &options).unwrap();

    // Corrupt the descriptor's NaN method name to a 7-char bogus
    // string by tampering with the CBOR.  We know the encoder
    // emitted `method: "roaring"` (7 ASCII chars) for the NaN
    // mask — replace the bytes in-place so the CBOR tstr(7) length
    // prefix stays valid.
    let roaring = b"roaring";
    let replacement = b"bogus!!"; // 7 chars — same length as "roaring"
    assert_eq!(replacement.len(), roaring.len());
    let needle_pos = msg
        .windows(roaring.len())
        .position(|w| w == roaring)
        .expect("encoded message should contain \"roaring\"");
    msg[needle_pos..needle_pos + roaring.len()].copy_from_slice(replacement);

    let err = decode(&msg, &DecodeOptions::default()).unwrap_err();
    let msg_str = err.to_string();
    assert!(
        msg_str.contains("unknown mask method"),
        "expected unknown-method error, got: {msg_str}"
    );
}

// ── Multiple objects per message, mixed finite / non-finite ─────────────────

#[test]
fn multi_object_message_with_mixed_finite_and_nan_payloads() {
    // Object 0: finite.  Object 1: has NaN.  Object 2: finite.
    // Only object 1 should have masks.
    let desc = make_descriptor(vec![3], Dtype::Float64);
    let d0 = f64_bytes(&[1.0, 2.0, 3.0]);
    let d1 = f64_bytes(&[f64::NAN, 4.0, 5.0]);
    let d2 = f64_bytes(&[6.0, 7.0, 8.0]);

    let options = EncodeOptions {
        allow_nan: true,
        hash_algorithm: None,
        small_mask_threshold_bytes: 0,
        ..Default::default()
    };
    let msg = encode(
        &make_global_meta(),
        &[(&desc, &d0), (&desc, &d1), (&desc, &d2)],
        &options,
    )
    .unwrap();
    let (_, descriptors) = decode_descriptors(&msg).unwrap();
    assert!(descriptors[0].masks.is_none());
    assert!(descriptors[1].masks.is_some());
    assert!(descriptors[2].masks.is_none());
}
