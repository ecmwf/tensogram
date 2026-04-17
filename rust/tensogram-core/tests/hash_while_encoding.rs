// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Integration tests for the hash-while-encoding optimisation.
//!
//! The optimisation folds xxh3-64 hashing into the encoding pipeline so
//! that the encoded payload is walked exactly once instead of twice
//! (once for encoding + framing, once for post-hoc `compute_hash`).
//! These tests guard the contract:
//!
//! 1. The descriptor hash attached by the buffered and streaming
//!    encoders equals `xxh3_64(encoded_payload)` formatted as a 16-char
//!    lowercase hex string.
//! 2. The buffered and streaming encoders attach the same hash for the
//!    same descriptor + payload.  Wire-format byte-identity across
//!    versions is covered separately by `golden_files.rs`.
//! 3. Multi-threaded transparent codecs produce the same hash at every
//!    thread count — the hash simply tracks the encoded bytes, which
//!    are byte-identical by contract.
//! 4. `hash_algorithm = None` produces no descriptor hash and no inline
//!    work (regression guard for "zero overhead when disabled").

use std::collections::BTreeMap;
use std::io::Cursor;

use tensogram_core::framing;
use tensogram_core::hash::{HashAlgorithm, compute_hash};
use tensogram_core::streaming::StreamingEncoder;
use tensogram_core::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata};
use tensogram_core::{DecodeOptions, Dtype, EncodeOptions, decode, encode};

// ── fixtures ──────────────────────────────────────────────────────────────────

fn make_desc_float32(n: usize) -> DataObjectDescriptor {
    DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![n as u64],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    }
}

fn payload_float32(n: usize, seed: u32) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            ((i as u32).wrapping_mul(0x9E37_79B1).wrapping_add(seed) as f32).to_ne_bytes()
        })
        .collect()
}

// ── hash-tracks-payload invariant ─────────────────────────────────────────────

#[test]
fn buffered_hash_equals_compute_hash_of_encoded_payload_passthrough() {
    // For a passthrough pipeline the encoded payload is byte-equal to the
    // input, so the descriptor hash must also equal xxh3_64(input).
    let meta = GlobalMetadata::default();
    let desc = make_desc_float32(1024);
    let data = payload_float32(1024, 0x1234_5678);

    let msg = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let decoded = framing::decode_message(&msg).unwrap();
    let (got_desc, got_payload, _) = &decoded.objects[0];

    let h = got_desc
        .hash
        .as_ref()
        .expect("EncodeOptions::default() requests xxh3");
    assert_eq!(h.hash_type, "xxh3");
    assert_eq!(h.value, compute_hash(got_payload, HashAlgorithm::Xxh3));
    assert_eq!(got_payload as &[u8], data.as_slice());
}

#[test]
fn streaming_hash_equals_compute_hash_of_encoded_payload() {
    let meta = GlobalMetadata::default();
    let desc = make_desc_float32(4096);
    let data = payload_float32(4096, 0xDEAD_BEEF);

    let buf = Vec::new();
    let mut enc = StreamingEncoder::new(buf, &meta, &EncodeOptions::default()).unwrap();
    enc.write_object(&desc, &data).unwrap();
    let msg = enc.finish().unwrap();

    let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    let (got_desc, got_payload) = &objects[0];

    let h = got_desc.hash.as_ref().expect("default options hash");
    assert_eq!(h.hash_type, "xxh3");
    assert_eq!(h.value, compute_hash(got_payload, HashAlgorithm::Xxh3));
}

#[test]
fn buffered_and_streaming_produce_identical_hashes() {
    // Given the same descriptor and bytes, both encode paths must attach
    // the same hash — the payload bytes going through simple_packing are
    // deterministic, and the hash is a pure function of those bytes.
    let meta = GlobalMetadata::default();

    let n = 2048;
    let values: Vec<f64> = (0..n).map(|i| 280.0 + (i as f64 * 0.01).sin()).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let params = tensogram_encodings::simple_packing::compute_params(&values, 24, 0).unwrap();

    let mut params_map: BTreeMap<String, ciborium::Value> = BTreeMap::new();
    params_map.insert(
        "reference_value".to_string(),
        ciborium::Value::Float(params.reference_value),
    );
    params_map.insert(
        "binary_scale_factor".to_string(),
        ciborium::Value::Integer((params.binary_scale_factor as i64).into()),
    );
    params_map.insert(
        "decimal_scale_factor".to_string(),
        ciborium::Value::Integer((params.decimal_scale_factor as i64).into()),
    );
    params_map.insert(
        "bits_per_value".to_string(),
        ciborium::Value::Integer((params.bits_per_value as i64).into()),
    );

    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![n as u64],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Little,
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: params_map,
        hash: None,
    };

    // Buffered.
    let msg_buf = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
    let decoded_buf = framing::decode_message(&msg_buf).unwrap();
    let buf_hash = decoded_buf.objects[0]
        .0
        .hash
        .as_ref()
        .unwrap()
        .value
        .clone();

    // Streaming.
    let sink = Vec::new();
    let mut enc = StreamingEncoder::new(sink, &meta, &EncodeOptions::default()).unwrap();
    enc.write_object(&desc, &data).unwrap();
    let msg_stream = enc.finish().unwrap();
    let (_, objects) = decode(&msg_stream, &DecodeOptions::default()).unwrap();
    let stream_hash = objects[0].0.hash.as_ref().unwrap().value.clone();

    assert_eq!(
        buf_hash, stream_hash,
        "buffered and streaming encoders must attach the same hash \
         for the same descriptor + payload"
    );
}

// ── no-hash path must attach no hash ──────────────────────────────────────────

#[test]
fn hash_algorithm_none_attaches_no_hash_buffered() {
    let meta = GlobalMetadata::default();
    let desc = make_desc_float32(64);
    let data = payload_float32(64, 0);
    let opts = EncodeOptions {
        hash_algorithm: None,
        ..Default::default()
    };
    let msg = encode(&meta, &[(&desc, &data)], &opts).unwrap();
    let decoded = framing::decode_message(&msg).unwrap();
    assert!(decoded.objects[0].0.hash.is_none());
}

#[test]
fn hash_algorithm_none_attaches_no_hash_streaming() {
    let meta = GlobalMetadata::default();
    let desc = make_desc_float32(64);
    let data = payload_float32(64, 0);
    let opts = EncodeOptions {
        hash_algorithm: None,
        ..Default::default()
    };
    let sink = Vec::new();
    let mut enc = StreamingEncoder::new(sink, &meta, &opts).unwrap();
    enc.write_object(&desc, &data).unwrap();
    let msg = enc.finish().unwrap();
    let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
    assert!(objects[0].0.hash.is_none());
}

// ── thread-count determinism ──────────────────────────────────────────────────

#[test]
fn buffered_hash_byte_identical_across_thread_counts_passthrough() {
    // Passthrough is transparent — bytes and therefore hashes must match
    // at every thread count.
    let meta = GlobalMetadata::default();
    let desc = make_desc_float32(128 * 1024);
    let data = payload_float32(128 * 1024, 0xCAFE_F00D);

    let mut hashes = Vec::new();
    for threads in [0u32, 1, 2, 4, 8] {
        let opts = EncodeOptions {
            threads,
            ..Default::default()
        };
        let msg = encode(&meta, &[(&desc, &data)], &opts).unwrap();
        let decoded = framing::decode_message(&msg).unwrap();
        hashes.push(decoded.objects[0].0.hash.as_ref().unwrap().value.clone());
    }
    for pair in hashes.windows(2) {
        assert_eq!(pair[0], pair[1], "hashes must be identical: {hashes:?}");
    }
}

#[test]
fn buffered_hash_byte_identical_across_thread_counts_simple_packing() {
    // simple_packing is transparent under threading — byte-identical
    // output → byte-identical hash.
    let meta = GlobalMetadata::default();
    let n = 32_768;
    let values: Vec<f64> = (0..n).map(|i| 280.0 + (i as f64 * 0.01).cos()).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let params = tensogram_encodings::simple_packing::compute_params(&values, 24, 0).unwrap();

    let mut params_map: BTreeMap<String, ciborium::Value> = BTreeMap::new();
    params_map.insert(
        "reference_value".to_string(),
        ciborium::Value::Float(params.reference_value),
    );
    params_map.insert(
        "binary_scale_factor".to_string(),
        ciborium::Value::Integer((params.binary_scale_factor as i64).into()),
    );
    params_map.insert(
        "decimal_scale_factor".to_string(),
        ciborium::Value::Integer((params.decimal_scale_factor as i64).into()),
    );
    params_map.insert(
        "bits_per_value".to_string(),
        ciborium::Value::Integer((params.bits_per_value as i64).into()),
    );

    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![n as u64],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Little,
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: params_map,
        hash: None,
    };

    let mut hashes = Vec::new();
    for threads in [0u32, 1, 2, 4] {
        let opts = EncodeOptions {
            threads,
            ..Default::default()
        };
        let msg = encode(&meta, &[(&desc, &data)], &opts).unwrap();
        let decoded = framing::decode_message(&msg).unwrap();
        hashes.push(decoded.objects[0].0.hash.as_ref().unwrap().value.clone());
    }
    for pair in hashes.windows(2) {
        assert_eq!(
            pair[0], pair[1],
            "simple_packing: hashes across thread counts must be byte-identical: {hashes:?}"
        );
    }
}

#[test]
fn streaming_hash_byte_identical_across_thread_counts_passthrough() {
    let meta = GlobalMetadata::default();
    let desc = make_desc_float32(32 * 1024);
    let data = payload_float32(32 * 1024, 0xAB_CDEF);

    let mut hashes = Vec::new();
    for threads in [0u32, 1, 2, 4] {
        let opts = EncodeOptions {
            threads,
            ..Default::default()
        };
        let sink = Cursor::new(Vec::new());
        let mut enc = StreamingEncoder::new(sink, &meta, &opts).unwrap();
        enc.write_object(&desc, &data).unwrap();
        let msg = enc.finish().unwrap().into_inner();
        let (_, objects) = decode(&msg, &DecodeOptions::default()).unwrap();
        hashes.push(objects[0].0.hash.as_ref().unwrap().value.clone());
    }
    for pair in hashes.windows(2) {
        assert_eq!(
            pair[0], pair[1],
            "streaming hashes must be identical: {hashes:?}"
        );
    }
}

// ── verify_hash round-trip for multi-object payloads ──────────────────────────

#[test]
fn decode_with_verify_hash_succeeds_for_multiple_objects() {
    let meta = GlobalMetadata::default();

    let desc1 = make_desc_float32(1024);
    let desc2 = make_desc_float32(512);
    let data1 = payload_float32(1024, 1);
    let data2 = payload_float32(512, 2);

    let msg = encode(
        &meta,
        &[(&desc1, &data1), (&desc2, &data2)],
        &EncodeOptions::default(),
    )
    .unwrap();

    let opts = DecodeOptions {
        verify_hash: true,
        ..Default::default()
    };
    let (_, objects) = decode(&msg, &opts).unwrap();
    assert_eq!(objects.len(), 2);
    assert_eq!(objects[0].1, data1);
    assert_eq!(objects[1].1, data2);
}

// ── framing-strategy divergence guard ────────────────────────────────────────

#[test]
fn buffered_and_streaming_wire_bytes_differ_by_design() {
    // Buffered and streaming encoders use different framing strategies —
    // header index + known `total_length` versus footer index + `0`
    // placeholder — so the two byte streams are NOT expected to match.
    //
    // The byte-identity we *do* guarantee (buffered output unchanged,
    // streaming output unchanged) is covered by `golden_files.rs`.  The
    // descriptor-level equivalence (same descriptor + hash on decode) is
    // covered by `buffered_and_streaming_produce_identical_hashes` above.
    //
    // This test documents the non-equivalence so a future change that
    // inadvertently made them byte-equal would be caught by CI rather than
    // silently altering the framing contract.
    let meta = GlobalMetadata::default();
    let desc = make_desc_float32(256);
    let data = payload_float32(256, 0);

    let buffered = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let sink = Vec::new();
    let mut enc = StreamingEncoder::new(sink, &meta, &EncodeOptions::default()).unwrap();
    enc.write_object(&desc, &data).unwrap();
    let streamed = enc.finish().unwrap();

    assert_ne!(buffered, streamed, "framing strategies differ by design");
}

// ── streaming CBOR-length invariant guard ────────────────────────────────────

/// The streaming encoder's inline-hash optimisation relies on the
/// `HashAlgorithm`-advertised hex-digest length to pre-size the CBOR
/// descriptor before the payload is hashed.  This test is a regression
/// guard: for every `HashAlgorithm` variant known to the library, every
/// reasonable descriptor shape must serialise to a CBOR byte-count that
/// is independent of the digest value.  The streaming path's pre-write
/// check enforces the same invariant — this test confirms no current
/// algorithm trips that check on the happy path.
#[test]
fn streaming_hash_algorithms_have_fixed_cbor_length() {
    use tensogram_core::metadata::object_descriptor_to_cbor;
    use tensogram_core::types::HashDescriptor;

    // Try a spread of descriptor shapes so we catch both "small CBOR"
    // and "large CBOR with many params" cases.
    let descriptors: Vec<DataObjectDescriptor> = vec![
        make_desc_float32(1),         // tiny shape
        make_desc_float32(1_000),     // moderate
        make_desc_float32(1_000_000), // large shape number (multi-byte varint)
    ];

    // Add every currently known HashAlgorithm here.  Adding a new
    // variant means adding it to this list AND confirming
    // `hex_digest_len` declares its fixed length correctly.
    for alg in [HashAlgorithm::Xxh3] {
        let hex_len = alg.hex_digest_len();
        for base_desc in &descriptors {
            let mut desc = base_desc.clone();
            desc.hash = Some(HashDescriptor {
                hash_type: alg.as_str().to_string(),
                value: "0".repeat(hex_len),
            });
            let len_zeros = object_descriptor_to_cbor(&desc).unwrap().len();

            desc.hash = Some(HashDescriptor {
                hash_type: alg.as_str().to_string(),
                value: "f".repeat(hex_len),
            });
            let len_ones = object_descriptor_to_cbor(&desc).unwrap().len();

            assert_eq!(
                len_zeros,
                len_ones,
                "{}: CBOR length must be independent of digest value \
                 (hex_digest_len={hex_len}, zeros={len_zeros}, \
                 ones={len_ones}).  The streaming encoder would reject \
                 this algorithm with a TensogramError::Framing.",
                alg.as_str()
            );
        }
    }
}
