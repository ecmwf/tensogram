//! Integration tests for `encode_pre_encoded()`.
//!
//! These tests cover happy-path round-trips, all encoding/compression
//! variants, edge cases, and rejection branches of the pre-encoded encode
//! API. Wire bytes are NOT compared directly because provenance fields
//! (`_reserved_.uuid`, `_reserved_.time`) are non-deterministic — instead
//! we compare a SHA-256 hash over (descriptor CBOR + decoded payload).
//!
//! Round-trip pattern:
//! 1. Encode raw data via `encode()`.
//! 2. Use `framing::decode_message` to extract the **encoded payload bytes**
//!    (the on-wire post-pipeline bytes — NOT the post-decode raw bytes).
//! 3. Feed those bytes back into `encode_pre_encoded()` along with the same
//!    descriptor.
//! 4. Decode both messages with `decode()` and compare via `decoded_sha256`.
//!
//! Both messages must produce identical decoded payloads since the library
//! deterministically hashes the same encoded bytes (`xxh3` → same hash) and
//! the same pipeline applied to the same encoded bytes yields the same
//! decoded payload.

use std::collections::BTreeMap;

use tensogram_core::framing;
use tensogram_core::{
    decode, decode_range, encode, encode_pre_encoded, ByteOrder, DataObjectDescriptor,
    DecodeOptions, Dtype, EncodeOptions, GlobalMetadata, HashDescriptor, StreamingEncoder,
};
use tensogram_encodings::simple_packing;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// SHA-256 hash of (CBOR-encoded descriptor || payload).
///
/// Used for comparing two decoded objects. We strip `_reserved_` from the
/// descriptor's `params` defensively in case the library ever adds it
/// there (currently it does not — `_reserved_` lives in metadata, not
/// descriptor — but hashing must be stable across encoder versions).
fn decoded_sha256(desc: &DataObjectDescriptor, payload: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut desc_clone = desc.clone();
    // Defensive: ensure no reserved keys leak into the hash.
    desc_clone.params.remove("_reserved_");
    let mut cbor = Vec::new();
    ciborium::into_writer(&desc_clone, &mut cbor).expect("encode descriptor for hashing");
    hasher.update(&cbor);
    hasher.update(payload);
    hasher.finalize().into()
}

/// Convert a slice of `f64` to its big-endian byte representation.
fn f64_to_be_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_be_bytes()).collect()
}

/// Convert a slice of `f32` to its big-endian byte representation.
fn f32_to_be_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_be_bytes()).collect()
}

/// Build a `simple_packing`-only descriptor (no compression).
fn make_simple_packing_desc(
    num_values: u64,
    p: &simple_packing::SimplePackingParams,
) -> DataObjectDescriptor {
    let mut params = BTreeMap::new();
    params.insert(
        "reference_value".to_string(),
        ciborium::Value::Float(p.reference_value),
    );
    params.insert(
        "binary_scale_factor".to_string(),
        ciborium::Value::Integer((p.binary_scale_factor as i64).into()),
    );
    params.insert(
        "decimal_scale_factor".to_string(),
        ciborium::Value::Integer((p.decimal_scale_factor as i64).into()),
    );
    params.insert(
        "bits_per_value".to_string(),
        ciborium::Value::Integer((p.bits_per_value as i64).into()),
    );
    DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![num_values],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Big,
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params,
        hash: None,
    }
}

/// Build a `simple_packing` + `szip` descriptor.
fn make_szip_simple_packing_desc(
    num_values: u64,
    p: &simple_packing::SimplePackingParams,
) -> DataObjectDescriptor {
    let mut params = BTreeMap::new();
    params.insert(
        "reference_value".to_string(),
        ciborium::Value::Float(p.reference_value),
    );
    params.insert(
        "binary_scale_factor".to_string(),
        ciborium::Value::Integer((p.binary_scale_factor as i64).into()),
    );
    params.insert(
        "decimal_scale_factor".to_string(),
        ciborium::Value::Integer((p.decimal_scale_factor as i64).into()),
    );
    params.insert(
        "bits_per_value".to_string(),
        ciborium::Value::Integer((p.bits_per_value as i64).into()),
    );
    params.insert("szip_rsi".to_string(), ciborium::Value::Integer(128.into()));
    params.insert(
        "szip_block_size".to_string(),
        ciborium::Value::Integer(16.into()),
    );
    params.insert(
        "szip_flags".to_string(),
        ciborium::Value::Integer(8_i64.into()),
    );
    DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![num_values],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Big,
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "szip".to_string(),
        params,
        hash: None,
    }
}

/// Build a raw (encoding=none) descriptor with the given compression name and params.
fn make_raw_desc(
    num_values: u64,
    dtype: Dtype,
    compression: &str,
    params: BTreeMap<String, ciborium::Value>,
) -> DataObjectDescriptor {
    DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![num_values],
        strides: vec![1],
        dtype,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: compression.to_string(),
        params,
        hash: None,
    }
}

/// Encode raw bytes via `encode()`, then return the (descriptor, encoded payload bytes)
/// pair from the resulting wire message. The payload is the on-wire post-pipeline bytes
/// suitable for feeding back into `encode_pre_encoded`.
fn encode_then_extract_payload(
    meta: &GlobalMetadata,
    desc: &DataObjectDescriptor,
    raw: &[u8],
    options: &EncodeOptions,
) -> (Vec<u8>, DataObjectDescriptor, Vec<u8>) {
    let msg = encode(meta, &[(desc, raw)], options).expect("encode raw");
    // Materialize descriptor + payload into owned values so the borrow on `msg`
    // is released before we move `msg` into the return tuple.
    let (extracted_desc, payload_vec) = {
        let dec = framing::decode_message(&msg).expect("decode message");
        assert_eq!(dec.objects.len(), 1);
        let (d, payload_slice, _offset) = &dec.objects[0];
        (d.clone(), payload_slice.to_vec())
    };
    (msg, extracted_desc, payload_vec)
}

/// Run a full pre-encoded round-trip and return both decoded results
/// for hash comparison.
fn round_trip_via_pre_encoded(
    meta: &GlobalMetadata,
    desc: &DataObjectDescriptor,
    raw: &[u8],
) -> (DataObjectDescriptor, Vec<u8>, DataObjectDescriptor, Vec<u8>) {
    let opts = EncodeOptions::default();
    let (msg1, extracted_desc, encoded_payload) =
        encode_then_extract_payload(meta, desc, raw, &opts);

    // Decode msg1 (the original encoded message)
    let (_, decoded1) = decode(&msg1, &DecodeOptions::default()).expect("decode msg1");
    let (d1, p1) = decoded1.into_iter().next().expect("at least one object");

    // Re-encode via pre-encoded path with the extracted encoded payload bytes.
    let msg2 = encode_pre_encoded(meta, &[(&extracted_desc, &encoded_payload)], &opts)
        .expect("encode_pre_encoded");

    // Decode msg2
    let (_, decoded2) = decode(&msg2, &DecodeOptions::default()).expect("decode msg2");
    let (d2, p2) = decoded2.into_iter().next().expect("at least one object");

    (d1, p1, d2, p2)
}

// ── Round-trip tests ─────────────────────────────────────────────────────────

#[test]
fn test_encode_pre_encoded_roundtrip_simple_packing() {
    let values: Vec<f64> = (0..1024).map(|i| 250.0 + i as f64 * 0.01).collect();
    let raw = f64_to_be_bytes(&values);
    let p = simple_packing::compute_params(&values, 16, 0).expect("compute simple_packing");
    let desc = make_simple_packing_desc(1024, &p);
    let meta = GlobalMetadata::default();

    let (d1, p1, d2, p2) = round_trip_via_pre_encoded(&meta, &desc, &raw);

    let h1 = decoded_sha256(&d1, &p1);
    let h2 = decoded_sha256(&d2, &p2);
    assert_eq!(h1, h2, "decoded payloads must match (simple_packing)");
}

#[test]
fn test_encode_pre_encoded_roundtrip_simple_packing_szip() {
    let values: Vec<f64> = (0..4096).map(|i| 250.0 + i as f64 * 0.1).collect();
    let raw = f64_to_be_bytes(&values);
    let p = simple_packing::compute_params(&values, 16, 0).expect("compute simple_packing");
    let desc = make_szip_simple_packing_desc(4096, &p);
    let meta = GlobalMetadata::default();

    let (d1, p1, d2, p2) = round_trip_via_pre_encoded(&meta, &desc, &raw);

    // szip_block_offsets must survive in d2 — they were carried forward from
    // d1 (which received them from the raw encode pipeline) into the
    // pre-encoded path and back through decode.
    assert!(
        d2.params.contains_key("szip_block_offsets"),
        "szip_block_offsets must survive in pre-encoded re-decoded descriptor",
    );

    let h1 = decoded_sha256(&d1, &p1);
    let h2 = decoded_sha256(&d2, &p2);
    assert_eq!(h1, h2, "decoded payloads must match (simple_packing+szip)");
}

// IMPORTANT: `validate_object()` enforces `bytes_len == shape * dtype`
// when `encoding == "none"`, regardless of compression. This means
// `encode_pre_encoded` with raw (encoding="none") + non-trivial
// compression cannot accept already-compressed bytes — the size check
// will fail. To exercise pre-encoded with these compressions, we use
// `simple_packing` as the encoding wrapper, which bypasses the size
// check (per the inherited invariant in encode.rs).

/// Build a `simple_packing` + `<compression>` descriptor with raw szip-style
/// rsi/block_size/flags omitted (so build_pipeline_config goes through the
/// requested compression branch).
fn make_simple_packing_compressed_desc(
    num_values: u64,
    p: &simple_packing::SimplePackingParams,
    compression: &str,
    extra_params: BTreeMap<String, ciborium::Value>,
) -> DataObjectDescriptor {
    let mut desc = make_simple_packing_desc(num_values, p);
    desc.compression = compression.to_string();
    for (k, v) in extra_params {
        desc.params.insert(k, v);
    }
    desc
}

#[cfg(feature = "zstd")]
#[test]
fn test_encode_pre_encoded_roundtrip_zstd() {
    let values: Vec<f64> = (0..1024).map(|i| 200.0 + i as f64 * 0.01).collect();
    let raw = f64_to_be_bytes(&values);
    let p = simple_packing::compute_params(&values, 16, 0).expect("compute simple_packing");
    let mut extra = BTreeMap::new();
    extra.insert(
        "zstd_level".to_string(),
        ciborium::Value::Integer(3_i64.into()),
    );
    let desc = make_simple_packing_compressed_desc(1024, &p, "zstd", extra);
    let meta = GlobalMetadata::default();

    let (d1, p1, d2, p2) = round_trip_via_pre_encoded(&meta, &desc, &raw);

    let h1 = decoded_sha256(&d1, &p1);
    let h2 = decoded_sha256(&d2, &p2);
    assert_eq!(h1, h2, "decoded payloads must match (simple_packing+zstd)");
}

#[cfg(feature = "lz4")]
#[test]
fn test_encode_pre_encoded_roundtrip_lz4() {
    let values: Vec<f64> = (0..1024).map(|i| 100.0 + i as f64 * 0.5).collect();
    let raw = f64_to_be_bytes(&values);
    let p = simple_packing::compute_params(&values, 16, 0).expect("compute simple_packing");
    let desc = make_simple_packing_compressed_desc(1024, &p, "lz4", BTreeMap::new());
    let meta = GlobalMetadata::default();

    let (d1, p1, d2, p2) = round_trip_via_pre_encoded(&meta, &desc, &raw);

    let h1 = decoded_sha256(&d1, &p1);
    let h2 = decoded_sha256(&d2, &p2);
    assert_eq!(h1, h2, "decoded payloads must match (simple_packing+lz4)");
}

#[cfg(feature = "blosc2")]
#[test]
fn test_encode_pre_encoded_roundtrip_blosc2() {
    let values: Vec<f64> = (0..1024).map(|i| 50.0 + i as f64 * 0.25).collect();
    let raw = f64_to_be_bytes(&values);
    let p = simple_packing::compute_params(&values, 16, 0).expect("compute simple_packing");
    let mut extra = BTreeMap::new();
    extra.insert(
        "blosc2_codec".to_string(),
        ciborium::Value::Text("zstd".to_string()),
    );
    extra.insert(
        "blosc2_clevel".to_string(),
        ciborium::Value::Integer(5_i64.into()),
    );
    let desc = make_simple_packing_compressed_desc(1024, &p, "blosc2", extra);
    let meta = GlobalMetadata::default();

    let (d1, p1, d2, p2) = round_trip_via_pre_encoded(&meta, &desc, &raw);

    let h1 = decoded_sha256(&d1, &p1);
    let h2 = decoded_sha256(&d2, &p2);
    assert_eq!(
        h1, h2,
        "decoded payloads must match (simple_packing+blosc2)"
    );
}

// zfp and sz3 expect raw float buffers and do not compose with simple_packing
// (which produces packed integer bit-streams). We document the API constraint
// instead: passing already-compressed zfp/sz3 bytes via encode_pre_encoded
// with `encoding="none"` is rejected by `validate_object` because the
// compressed byte length cannot equal `shape * dtype` for non-trivial inputs.

#[cfg(feature = "zfp")]
#[test]
fn test_encode_pre_encoded_roundtrip_zfp_fixed_rate() {
    let values: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01).collect();
    let raw = f32_to_be_bytes(&values);
    let mut params = BTreeMap::new();
    params.insert(
        "zfp_mode".to_string(),
        ciborium::Value::Text("fixed_rate".to_string()),
    );
    params.insert("zfp_rate".to_string(), ciborium::Value::Float(8.0));
    let desc = make_raw_desc(1024, Dtype::Float32, "zfp", params);
    let meta = GlobalMetadata::default();
    let opts = EncodeOptions::default();

    // Get the actually-compressed bytes from a normal encode.
    let (_msg, extracted_desc, encoded_payload) =
        encode_then_extract_payload(&meta, &desc, &raw, &opts);
    // Pre-encoded with the same encoding="none" descriptor must FAIL because
    // the compressed payload length will not equal raw shape*dtype bytes.
    let result = encode_pre_encoded(&meta, &[(&extracted_desc, &encoded_payload)], &opts);
    assert!(
        result.is_err(),
        "pre-encoded with encoding=none + compressed bytes must be rejected"
    );
    let err = result.expect_err("err").to_string();
    assert!(
        err.contains("does not match expected"),
        "error should mention size mismatch, got: {err}"
    );
}

#[cfg(feature = "sz3")]
#[test]
fn test_encode_pre_encoded_roundtrip_sz3_abs() {
    let values: Vec<f32> = (0..1024).map(|i| 100.0 + (i as f32) * 0.1).collect();
    let raw = f32_to_be_bytes(&values);
    let mut params = BTreeMap::new();
    params.insert(
        "sz3_error_bound_mode".to_string(),
        ciborium::Value::Text("abs".to_string()),
    );
    params.insert("sz3_error_bound".to_string(), ciborium::Value::Float(0.01));
    let desc = make_raw_desc(1024, Dtype::Float32, "sz3", params);
    let meta = GlobalMetadata::default();
    let opts = EncodeOptions::default();

    // Same constraint as zfp: compressed sz3 bytes do not match raw size,
    // and encoding="none" path rejects them.
    let (_msg, extracted_desc, encoded_payload) =
        encode_then_extract_payload(&meta, &desc, &raw, &opts);
    let result = encode_pre_encoded(&meta, &[(&extracted_desc, &encoded_payload)], &opts);
    assert!(
        result.is_err(),
        "pre-encoded with encoding=none + sz3-compressed bytes must be rejected"
    );
    let err = result.expect_err("err").to_string();
    assert!(
        err.contains("does not match expected"),
        "error should mention size mismatch, got: {err}"
    );
}

// ── decode_range with caller-provided szip_block_offsets ─────────────────────

#[cfg(feature = "szip")]
#[test]
fn test_encode_pre_encoded_with_szip_decode_range() {
    let values: Vec<f64> = (0..4096).map(|i| 100.0 + i as f64 * 0.5).collect();
    let raw = f64_to_be_bytes(&values);
    let p = simple_packing::compute_params(&values, 16, 0).expect("compute simple_packing");
    let desc = make_szip_simple_packing_desc(4096, &p);
    let meta = GlobalMetadata::default();

    // Round-trip via pre-encoded path with offsets carried over.
    let opts = EncodeOptions::default();
    let (_msg1, extracted_desc, encoded_payload) =
        encode_then_extract_payload(&meta, &desc, &raw, &opts);
    assert!(
        extracted_desc.params.contains_key("szip_block_offsets"),
        "raw encode of szip must populate szip_block_offsets"
    );

    let msg2 = encode_pre_encoded(&meta, &[(&extracted_desc, &encoded_payload)], &opts)
        .expect("encode_pre_encoded");

    // Full decode of the pre-encoded message
    let (_, full_objects) = decode(&msg2, &DecodeOptions::default()).expect("decode full");
    let full_payload = &full_objects[0].1;
    let full_values: Vec<f64> = full_payload
        .chunks_exact(8)
        .map(|c| f64::from_be_bytes(c.try_into().expect("8 bytes")))
        .collect();
    assert_eq!(full_values.len(), 4096);

    // Partial decode: 500 elements starting at index 100.
    let parts =
        decode_range(&msg2, 0, &[(100, 500)], &DecodeOptions::default()).expect("decode_range");
    assert_eq!(parts.len(), 1, "one range → one part");
    let part_values: Vec<f64> = parts[0]
        .chunks_exact(8)
        .map(|c| f64::from_be_bytes(c.try_into().expect("8 bytes")))
        .collect();
    assert_eq!(part_values.len(), 500);

    // Compare against slice of full decode (within simple_packing tolerance).
    for (i, (full, partial)) in full_values[100..600]
        .iter()
        .zip(part_values.iter())
        .enumerate()
    {
        assert!(
            (full - partial).abs() < 0.01,
            "value mismatch at offset {i}: full={full}, partial={partial}"
        );
    }
}

#[cfg(feature = "szip")]
#[test]
fn test_encode_pre_encoded_decode_range_fails_without_offsets() {
    let values: Vec<f64> = (0..4096).map(|i| 50.0 + i as f64 * 0.25).collect();
    let raw = f64_to_be_bytes(&values);
    let p = simple_packing::compute_params(&values, 16, 0).expect("compute simple_packing");
    let desc = make_szip_simple_packing_desc(4096, &p);
    let meta = GlobalMetadata::default();
    let opts = EncodeOptions::default();

    // Encode raw to obtain valid szip-compressed payload bytes.
    let (_msg, mut extracted_desc, encoded_payload) =
        encode_then_extract_payload(&meta, &desc, &raw, &opts);
    // Strip the offsets — the pre-encoded path allows szip without offsets.
    extracted_desc.params.remove("szip_block_offsets");
    assert!(!extracted_desc.params.contains_key("szip_block_offsets"));

    let msg2 = encode_pre_encoded(&meta, &[(&extracted_desc, &encoded_payload)], &opts)
        .expect("encode_pre_encoded must succeed without offsets");

    // Full decode still works (it doesn't need block offsets).
    let _ = decode(&msg2, &DecodeOptions::default()).expect("full decode should succeed");

    // But decode_range must fail because it requires offsets.
    let result = decode_range(&msg2, 0, &[(0, 100)], &DecodeOptions::default());
    assert!(
        result.is_err(),
        "decode_range without szip_block_offsets must fail"
    );
    let err = result.expect_err("must err").to_string();
    assert!(
        err.contains("szip_block_offsets"),
        "error should mention szip_block_offsets, got: {err}"
    );
}

// ── Hash overwrite ───────────────────────────────────────────────────────────

#[test]
fn test_encode_pre_encoded_overwrites_caller_hash() {
    let values: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let raw = f32_to_be_bytes(&values);
    let garbage_hash = HashDescriptor {
        hash_type: "xxh3".to_string(),
        value: "deadbeefcafebabe".to_string(),
    };
    let mut desc = make_raw_desc(256, Dtype::Float32, "none", BTreeMap::new());
    desc.hash = Some(garbage_hash);

    let meta = GlobalMetadata::default();
    let opts = EncodeOptions::default();
    let msg = encode_pre_encoded(&meta, &[(&desc, &raw)], &opts).expect("encode_pre_encoded");

    let (_, objects) = decode(&msg, &DecodeOptions::default()).expect("decode");
    let embedded = objects[0].0.hash.as_ref().expect("hash present");
    assert_eq!(embedded.hash_type, "xxh3", "hash type should be xxh3");
    assert_ne!(
        embedded.value, "deadbeefcafebabe",
        "garbage hash must be overwritten by library"
    );

    // The library hashes the encoded payload bytes, which for encoding=none
    // are exactly the bytes the caller passed in.
    let expected = tensogram_core::compute_hash(&raw, tensogram_core::HashAlgorithm::Xxh3);
    assert_eq!(
        embedded.value, expected,
        "embedded hash must equal xxh3 of payload bytes"
    );
}

// ── Rejection branches ───────────────────────────────────────────────────────

#[test]
fn test_encode_pre_encoded_rejects_emit_preceders() {
    let raw = vec![0u8; 16]; // 4 × float32
    let desc = make_raw_desc(4, Dtype::Float32, "none", BTreeMap::new());
    let meta = GlobalMetadata::default();
    let opts = EncodeOptions {
        hash_algorithm: Some(tensogram_core::HashAlgorithm::Xxh3),
        emit_preceders: true,
        ..Default::default()
    };
    let result = encode_pre_encoded(&meta, &[(&desc, &raw)], &opts);
    assert!(result.is_err(), "emit_preceders=true must be rejected");
    let err = result.expect_err("err").to_string();
    assert!(
        err.contains("emit_preceders"),
        "error should mention emit_preceders, got: {err}"
    );
}

#[test]
fn test_encode_pre_encoded_rejects_caller_reserved() {
    let raw = vec![0u8; 16];
    let desc = make_raw_desc(4, Dtype::Float32, "none", BTreeMap::new());
    let mut reserved = BTreeMap::new();
    reserved.insert(
        "uuid".to_string(),
        ciborium::Value::Text("client-set".to_string()),
    );
    let meta = GlobalMetadata {
        version: 2,
        reserved,
        ..Default::default()
    };
    let result = encode_pre_encoded(&meta, &[(&desc, &raw)], &EncodeOptions::default());
    assert!(result.is_err(), "caller-set _reserved_ must be rejected");
    let err = result.expect_err("err").to_string();
    assert!(
        err.contains("_reserved_"),
        "error should mention _reserved_, got: {err}"
    );
}

#[test]
fn test_encode_pre_encoded_rejects_szip_offsets_for_non_szip() {
    let raw = vec![0u8; 16];
    let mut params = BTreeMap::new();
    // Provide szip_block_offsets but compression is zstd → must fail.
    params.insert(
        "szip_block_offsets".to_string(),
        ciborium::Value::Array(vec![ciborium::Value::Integer(0_i64.into())]),
    );
    params.insert(
        "zstd_level".to_string(),
        ciborium::Value::Integer(3_i64.into()),
    );
    let desc = make_raw_desc(4, Dtype::Float32, "zstd", params);
    let meta = GlobalMetadata::default();
    let result = encode_pre_encoded(&meta, &[(&desc, &raw)], &EncodeOptions::default());
    assert!(
        result.is_err(),
        "szip_block_offsets with non-szip compression must be rejected"
    );
    let err = result.expect_err("err").to_string();
    assert!(
        err.contains("szip_block_offsets"),
        "error should mention szip_block_offsets, got: {err}"
    );
}

#[cfg(feature = "szip")]
#[test]
fn test_encode_pre_encoded_rejects_non_monotonic_offsets() {
    // Build any szip descriptor (won't actually run szip pipeline since
    // pre-encoded mode skips encoding) and inject bad offsets.
    let dummy_raw = vec![0u8; 4096];
    let p = simple_packing::SimplePackingParams {
        reference_value: 0.0,
        binary_scale_factor: 0,
        decimal_scale_factor: 0,
        bits_per_value: 16,
    };
    let mut desc = make_szip_simple_packing_desc(512, &p);
    desc.params.insert(
        "szip_block_offsets".to_string(),
        ciborium::Value::Array(vec![
            ciborium::Value::Integer(0_i64.into()),
            ciborium::Value::Integer(100_i64.into()),
            ciborium::Value::Integer(50_i64.into()), // not strictly increasing
        ]),
    );
    let meta = GlobalMetadata::default();
    let result = encode_pre_encoded(&meta, &[(&desc, &dummy_raw)], &EncodeOptions::default());
    assert!(
        result.is_err(),
        "non-monotonic szip_block_offsets must be rejected"
    );
    let err = result.expect_err("err").to_string();
    assert!(
        err.contains("strictly increasing") || err.contains("szip_block_offsets"),
        "error should mention monotonicity, got: {err}"
    );
}

#[cfg(feature = "szip")]
#[test]
fn test_encode_pre_encoded_rejects_offset_beyond_buffer() {
    let dummy_raw = vec![0u8; 16]; // 16 bytes = 128 bits
    let p = simple_packing::SimplePackingParams {
        reference_value: 0.0,
        binary_scale_factor: 0,
        decimal_scale_factor: 0,
        bits_per_value: 16,
    };
    let mut desc = make_szip_simple_packing_desc(8, &p);
    // Bit-bound is 128; offset 999 must overflow it.
    desc.params.insert(
        "szip_block_offsets".to_string(),
        ciborium::Value::Array(vec![
            ciborium::Value::Integer(0_i64.into()),
            ciborium::Value::Integer(999_i64.into()),
        ]),
    );
    let meta = GlobalMetadata::default();
    let result = encode_pre_encoded(&meta, &[(&desc, &dummy_raw)], &EncodeOptions::default());
    assert!(result.is_err(), "offset beyond bit bound must be rejected");
    let err = result.expect_err("err").to_string();
    assert!(
        err.contains("bit bound") || err.contains("exceeds"),
        "error should mention bound violation, got: {err}"
    );
}

// ── Edge cases ───────────────────────────────────────────────────────────────

#[test]
fn test_encode_pre_encoded_zero_objects() {
    let meta = GlobalMetadata::default();
    let msg = encode_pre_encoded(&meta, &[], &EncodeOptions::default())
        .expect("zero-object encode_pre_encoded must succeed");
    let (decoded_meta, objects) =
        decode(&msg, &DecodeOptions::default()).expect("decode empty message");
    assert_eq!(objects.len(), 0, "should decode to zero objects");
    assert_eq!(decoded_meta.version, 2);
}

#[test]
fn test_encode_pre_encoded_zero_element_shape() {
    // shape [0, 5] → product 0 → expected_bytes 0 → empty payload is valid.
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 2,
        shape: vec![0, 5],
        strides: vec![5, 1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    };
    let meta = GlobalMetadata::default();
    let msg = encode_pre_encoded(&meta, &[(&desc, &[])], &EncodeOptions::default())
        .expect("zero-element shape must succeed");
    let (_, objects) = decode(&msg, &DecodeOptions::default()).expect("decode");
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].0.shape, vec![0, 5]);
    assert!(objects[0].1.is_empty(), "payload must be empty");
}

#[test]
fn test_encode_pre_encoded_provenance_populated() {
    let raw = vec![0u8; 16]; // 4 × float32
    let desc = make_raw_desc(4, Dtype::Float32, "none", BTreeMap::new());
    let meta = GlobalMetadata::default();
    let msg = encode_pre_encoded(&meta, &[(&desc, &raw)], &EncodeOptions::default())
        .expect("encode_pre_encoded");
    let (decoded_meta, _) = decode(&msg, &DecodeOptions::default()).expect("decode");

    // _reserved_ should have encoder, time, uuid populated.
    let reserved = &decoded_meta.reserved;
    assert!(reserved.contains_key("encoder"), "encoder missing");
    assert!(reserved.contains_key("time"), "time missing");
    assert!(reserved.contains_key("uuid"), "uuid missing");

    // encoder.name should be "tensogram"
    if let Some(ciborium::Value::Map(pairs)) = reserved.get("encoder") {
        let name_pair = pairs
            .iter()
            .find(|(k, _)| *k == ciborium::Value::Text("name".to_string()));
        assert!(
            matches!(name_pair, Some((_, ciborium::Value::Text(s))) if s == "tensogram"),
            "encoder.name must be 'tensogram'"
        );
    } else {
        panic!("encoder must be a map");
    }

    // time should be a non-empty string
    if let Some(ciborium::Value::Text(t)) = reserved.get("time") {
        assert!(!t.is_empty(), "time must not be empty");
    } else {
        panic!("time must be a string");
    }

    // uuid should be a non-empty string
    if let Some(ciborium::Value::Text(u)) = reserved.get("uuid") {
        assert!(!u.is_empty(), "uuid must not be empty");
    } else {
        panic!("uuid must be a string");
    }
}

#[test]
fn test_encode_pre_encoded_tensor_metadata_populated() {
    let raw = vec![0u8; 3 * 4 * 4]; // 12 × float32
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 2,
        shape: vec![3, 4],
        strides: vec![4, 1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    };
    let meta = GlobalMetadata::default();
    let msg = encode_pre_encoded(&meta, &[(&desc, &raw)], &EncodeOptions::default())
        .expect("encode_pre_encoded");
    let (decoded_meta, _) = decode(&msg, &DecodeOptions::default()).expect("decode");

    let base0 = &decoded_meta.base[0];
    let reserved = base0
        .get("_reserved_")
        .expect("_reserved_ missing in base[0]");
    if let ciborium::Value::Map(pairs) = reserved {
        let tensor_entry = pairs
            .iter()
            .find(|(k, _)| *k == ciborium::Value::Text("tensor".to_string()));
        let tensor_map = match tensor_entry {
            Some((_, ciborium::Value::Map(m))) => m,
            _ => panic!("tensor missing or not a map"),
        };
        let mut keys: Vec<String> = tensor_map
            .iter()
            .filter_map(|(k, _)| {
                if let ciborium::Value::Text(s) = k {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "dtype".to_string(),
                "ndim".to_string(),
                "shape".to_string(),
                "strides".to_string(),
            ]
        );
        // Verify dtype is float32
        let dtype_pair = tensor_map
            .iter()
            .find(|(k, _)| *k == ciborium::Value::Text("dtype".to_string()));
        assert!(
            matches!(dtype_pair, Some((_, ciborium::Value::Text(s))) if s == "float32"),
            "dtype must be float32"
        );
    } else {
        panic!("_reserved_ must be a map");
    }
}

// ── Streaming integration ────────────────────────────────────────────────────

#[test]
fn test_streaming_mixed_mode_pre_encoded() {
    // Streaming: write_object (raw), write_object_pre_encoded, write_object (raw).
    let meta = GlobalMetadata::default();
    let opts = EncodeOptions::default();

    let desc0 = make_raw_desc(4, Dtype::Float32, "none", BTreeMap::new());
    let desc1 = make_raw_desc(5, Dtype::Float32, "none", BTreeMap::new());
    let desc2 = make_raw_desc(6, Dtype::Float32, "none", BTreeMap::new());

    let data0 = vec![1u8; 4 * 4];
    let pre_encoded1 = vec![2u8; 5 * 4]; // bytes are treated as already-encoded
    let data2 = vec![3u8; 6 * 4];

    let buf: Vec<u8> = Vec::new();
    let mut enc = StreamingEncoder::new(buf, &meta, &opts).expect("create streaming encoder");
    enc.write_object(&desc0, &data0).expect("write 0 (raw)");
    enc.write_object_pre_encoded(&desc1, &pre_encoded1)
        .expect("write 1 (pre-encoded)");
    enc.write_object(&desc2, &data2).expect("write 2 (raw)");
    let result = enc.finish().expect("finish");

    let (_, objects) = decode(&result, &DecodeOptions::default()).expect("decode streaming");
    assert_eq!(objects.len(), 3, "must decode 3 objects");
    assert_eq!(objects[0].1, data0, "object 0 raw payload mismatch");
    assert_eq!(
        objects[1].1, pre_encoded1,
        "object 1 pre-encoded payload mismatch"
    );
    assert_eq!(objects[2].1, data2, "object 2 raw payload mismatch");
}

// ── Additional edge-case tests ───────────────────────────────────────────────

#[test]
fn test_encode_pre_encoded_single_element() {
    // Shape=[1]: single-element array round-trip.
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![1],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    };
    let raw = 42.0f32.to_be_bytes().to_vec();
    let meta = GlobalMetadata::default();
    let msg = encode_pre_encoded(&meta, &[(&desc, &raw)], &EncodeOptions::default())
        .expect("single element encode_pre_encoded");
    let (_, objects) = decode(&msg, &DecodeOptions::default()).expect("decode");
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].0.shape, vec![1]);
    let val = f32::from_be_bytes(objects[0].1[..4].try_into().unwrap());
    assert!((val - 42.0).abs() < f32::EPSILON, "value mismatch: {val}");
}

#[test]
fn test_encode_pre_encoded_2d_array() {
    // 2D shape [3, 4] encoding=none round-trip.
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 2,
        shape: vec![3, 4],
        strides: vec![4, 1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    };
    let values: Vec<f32> = (0..12).map(|i| i as f32 * 1.5).collect();
    let raw = f32_to_be_bytes(&values);
    let meta = GlobalMetadata::default();
    let msg = encode_pre_encoded(&meta, &[(&desc, &raw)], &EncodeOptions::default())
        .expect("2D encode_pre_encoded");
    let (_, objects) = decode(&msg, &DecodeOptions::default()).expect("decode");
    assert_eq!(objects[0].0.shape, vec![3, 4]);
    assert_eq!(objects[0].0.ndim, 2);
    assert_eq!(objects[0].1, raw, "2D payload round-trip");
}

#[test]
fn test_encode_pre_encoded_ndim0_scalar() {
    // ndim=0 scalar: shape=[], strides=[].
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 0,
        shape: vec![],
        strides: vec![],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    };
    // A scalar has shape product = 1 (empty product), so expected bytes = 1 * 8 = 8.
    let raw = std::f64::consts::PI.to_be_bytes().to_vec();
    let meta = GlobalMetadata::default();
    let msg = encode_pre_encoded(&meta, &[(&desc, &raw)], &EncodeOptions::default())
        .expect("scalar encode_pre_encoded");
    let (_, objects) = decode(&msg, &DecodeOptions::default()).expect("decode");
    assert_eq!(objects[0].0.ndim, 0);
    assert!(objects[0].0.shape.is_empty());
    let val = f64::from_be_bytes(objects[0].1[..8].try_into().unwrap());
    assert!((val - std::f64::consts::PI).abs() < f64::EPSILON);
}

#[test]
fn test_encode_pre_encoded_rejects_empty_obj_type() {
    let desc = DataObjectDescriptor {
        obj_type: "".to_string(),
        ndim: 1,
        shape: vec![4],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    };
    let raw = vec![0u8; 16];
    let meta = GlobalMetadata::default();
    let result = encode_pre_encoded(&meta, &[(&desc, &raw)], &EncodeOptions::default());
    assert!(result.is_err(), "empty obj_type must be rejected");
    let err = result.expect_err("err").to_string();
    assert!(
        err.contains("obj_type"),
        "error should mention obj_type, got: {err}"
    );
}

#[test]
fn test_encode_pre_encoded_encoding_none_data_too_short() {
    // encoding=none with data shorter than shape*dtype → rejected.
    let desc = make_raw_desc(10, Dtype::Float32, "none", BTreeMap::new());
    let raw = vec![0u8; 20]; // 20 bytes, need 40 for 10 × float32
    let meta = GlobalMetadata::default();
    let result = encode_pre_encoded(&meta, &[(&desc, &raw)], &EncodeOptions::default());
    assert!(result.is_err(), "data too short must be rejected");
    let err = result.expect_err("err").to_string();
    assert!(
        err.contains("does not match expected"),
        "error should mention size mismatch, got: {err}"
    );
}

#[test]
fn test_encode_pre_encoded_encoding_none_data_too_long() {
    // encoding=none with data longer than shape*dtype → rejected.
    let desc = make_raw_desc(4, Dtype::Float32, "none", BTreeMap::new());
    let raw = vec![0u8; 32]; // 32 bytes, need 16 for 4 × float32
    let meta = GlobalMetadata::default();
    let result = encode_pre_encoded(&meta, &[(&desc, &raw)], &EncodeOptions::default());
    assert!(result.is_err(), "data too long must be rejected");
    let err = result.expect_err("err").to_string();
    assert!(
        err.contains("does not match expected"),
        "error should mention size mismatch, got: {err}"
    );
}

#[cfg(feature = "szip")]
#[test]
fn test_encode_pre_encoded_szip_single_offset() {
    // szip_block_offsets = [0] (single entry) should be accepted by encode.
    // NOTE: we do NOT decode because the dummy payload is not valid szip data.
    let p = simple_packing::SimplePackingParams {
        reference_value: 0.0,
        binary_scale_factor: 0,
        decimal_scale_factor: 0,
        bits_per_value: 16,
    };
    let mut desc = make_szip_simple_packing_desc(64, &p);
    desc.params.insert(
        "szip_block_offsets".to_string(),
        ciborium::Value::Array(vec![ciborium::Value::Integer(0_i64.into())]),
    );
    let dummy = vec![0u8; 128]; // some payload bytes
    let meta = GlobalMetadata::default();
    let _msg = encode_pre_encoded(&meta, &[(&desc, &dummy)], &EncodeOptions::default())
        .expect("single offset [0] must succeed");
    // Encode succeeded — structural validation passed.
}

#[cfg(feature = "szip")]
#[test]
fn test_encode_pre_encoded_szip_offset_at_exact_bit_boundary() {
    // Offset at exactly bytes_len * 8 should be accepted (boundary case).
    // NOTE: we do NOT decode because the dummy payload is not valid szip data.
    let p = simple_packing::SimplePackingParams {
        reference_value: 0.0,
        binary_scale_factor: 0,
        decimal_scale_factor: 0,
        bits_per_value: 16,
    };
    let dummy = vec![0u8; 32]; // 32 bytes = 256 bits
    let mut desc = make_szip_simple_packing_desc(16, &p);
    desc.params.insert(
        "szip_block_offsets".to_string(),
        ciborium::Value::Array(vec![
            ciborium::Value::Integer(0_i64.into()),
            ciborium::Value::Integer(128_i64.into()),
            ciborium::Value::Integer(256_i64.into()), // exactly at boundary
        ]),
    );
    let meta = GlobalMetadata::default();
    let _msg = encode_pre_encoded(&meta, &[(&desc, &dummy)], &EncodeOptions::default())
        .expect("offset at exact bit boundary must succeed");
    // Encode succeeded — structural validation passed.
}

#[test]
fn test_encode_pre_encoded_extra_params_survive() {
    // Unknown params in the descriptor should survive round-trip.
    let mut params = BTreeMap::new();
    params.insert(
        "custom_key".to_string(),
        ciborium::Value::Text("custom_value".to_string()),
    );
    params.insert(
        "numeric_param".to_string(),
        ciborium::Value::Integer(42_i64.into()),
    );
    let desc = make_raw_desc(4, Dtype::Float32, "none", params);
    let raw = vec![0u8; 16];
    let meta = GlobalMetadata::default();
    let msg = encode_pre_encoded(&meta, &[(&desc, &raw)], &EncodeOptions::default())
        .expect("extra params must succeed");
    let (_, objects) = decode(&msg, &DecodeOptions::default()).expect("decode");
    let out_params = &objects[0].0.params;
    assert_eq!(
        out_params.get("custom_key"),
        Some(&ciborium::Value::Text("custom_value".to_string())),
    );
    assert_eq!(
        out_params.get("numeric_param"),
        Some(&ciborium::Value::Integer(42_i64.into())),
    );
}

#[test]
fn test_encode_pre_encoded_no_hash() {
    // hash_algorithm=None: no hash in output.
    let raw = vec![0u8; 16];
    let desc = make_raw_desc(4, Dtype::Float32, "none", BTreeMap::new());
    let meta = GlobalMetadata::default();
    let opts = EncodeOptions {
        hash_algorithm: None,
        emit_preceders: false,
        ..Default::default()
    };
    let msg = encode_pre_encoded(&meta, &[(&desc, &raw)], &opts)
        .expect("encode_pre_encoded with no hash");
    let (_, objects) = decode(&msg, &DecodeOptions::default()).expect("decode");
    assert!(objects[0].0.hash.is_none(), "hash must be None");
}

#[test]
fn test_encode_pre_encoded_multiple_objects_different_dtypes() {
    // Two objects with different dtypes in one message.
    let desc_f32 = make_raw_desc(4, Dtype::Float32, "none", BTreeMap::new());
    let desc_f64 = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![3],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    };
    let raw_f32 = vec![1u8; 16]; // 4 × float32
    let raw_f64 = vec![2u8; 24]; // 3 × float64
    let meta = GlobalMetadata::default();
    let msg = encode_pre_encoded(
        &meta,
        &[(&desc_f32, &raw_f32[..]), (&desc_f64, &raw_f64[..])],
        &EncodeOptions::default(),
    )
    .expect("multi-dtype encode_pre_encoded");
    let (_, objects) = decode(&msg, &DecodeOptions::default()).expect("decode");
    assert_eq!(objects.len(), 2);
    assert_eq!(objects[0].0.dtype, Dtype::Float32);
    assert_eq!(objects[1].0.dtype, Dtype::Float64);
    assert_eq!(objects[0].1, raw_f32);
    assert_eq!(objects[1].1, raw_f64);
}

#[test]
fn test_encode_pre_encoded_ndim_shape_mismatch_rejected() {
    // ndim=2 but shape has 1 element → rejected.
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 2,
        shape: vec![4],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    };
    let raw = vec![0u8; 16];
    let meta = GlobalMetadata::default();
    let result = encode_pre_encoded(&meta, &[(&desc, &raw)], &EncodeOptions::default());
    assert!(result.is_err(), "ndim/shape mismatch must be rejected");
    let err = result.expect_err("err").to_string();
    assert!(
        err.contains("ndim") && err.contains("shape"),
        "error should mention ndim/shape mismatch, got: {err}"
    );
}

#[test]
fn test_encode_pre_encoded_strides_shape_mismatch_rejected() {
    // strides.len() != shape.len() → rejected.
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![4],
        strides: vec![1, 1], // wrong length
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    };
    let raw = vec![0u8; 16];
    let meta = GlobalMetadata::default();
    let result = encode_pre_encoded(&meta, &[(&desc, &raw)], &EncodeOptions::default());
    assert!(result.is_err(), "strides/shape mismatch must be rejected");
    let err = result.expect_err("err").to_string();
    assert!(
        err.contains("strides") && err.contains("shape"),
        "error should mention strides/shape mismatch, got: {err}"
    );
}

#[test]
fn test_streaming_pre_encoded_with_preceder() {
    // Streaming: write_object_pre_encoded after writing a preceder.
    let meta = GlobalMetadata::default();
    let opts = EncodeOptions {
        hash_algorithm: Some(tensogram_core::HashAlgorithm::Xxh3),
        emit_preceders: true,
        ..Default::default()
    };

    let desc = make_raw_desc(4, Dtype::Float32, "none", BTreeMap::new());
    let raw = vec![42u8; 16]; // 4 × float32

    let buf: Vec<u8> = Vec::new();
    let mut enc = StreamingEncoder::new(buf, &meta, &opts).expect("create streaming encoder");

    // Write a preceder (metadata-only, no data payload).
    let preceder_meta: BTreeMap<String, ciborium::Value> = BTreeMap::new();
    enc.write_preceder(preceder_meta).expect("write preceder");

    // Then write the pre-encoded object
    enc.write_object_pre_encoded(&desc, &raw)
        .expect("write pre-encoded after preceder");

    let result = enc.finish().expect("finish");
    let (_, objects) = decode(&result, &DecodeOptions::default()).expect("decode");
    // Preceder is transparent to decode — only the main object is returned.
    assert_eq!(
        objects.len(),
        1,
        "preceder + 1 pre-encoded object → 1 decoded"
    );
    assert_eq!(objects[0].1, raw);
}
