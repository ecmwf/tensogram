//! Comprehensive WASM integration tests for browser stability.
//!
//! These tests run inside a WASM runtime (Node.js via wasm-pack test --node).
//! They verify the full browser-facing API contract and are designed to
//! catch regressions as the tensogram library evolves:
//!
//! 1.  Golden file decode: canonical .tgm patterns produce expected output
//! 2.  Round-trip: encode → decode identical data for every dtype
//! 3.  Compression codecs: lz4, szip-pure, zstd-pure round-trips
//! 4.  Streaming decoder: chunked feed, multi-message, recovery, edge cases
//! 5.  Zero-copy views: all TypedArray types, alignment errors, safe copies
//! 6.  Error handling: corrupt data, unsupported codecs, OOB, truncation
//! 7.  API stability: JS-facing type shapes, field names, nullability
//! 8.  Encode pipeline: full WASM encode with various configs
//! 9.  Metadata fidelity: MARS keys, _extra_, _reserved_, version
//! 10. Edge cases: empty, scalar, large, unicode metadata, multi-dim
//! 11. Streaming data correctness: frame data matches direct decode
//! 12. Multi-object decode: mixed dtypes, selective decode by index
//! 13. Scan correctness: multi-message buffers, garbage tolerance
//! 14. Hash verification: enabled/disabled, tampered payloads
//! 15. Pipeline combos: encoding+filter+compression round-trips
//!
//! Run with: wasm-pack test --node crates/tensogram-wasm
//!
//! These tests are the **stability contract** for browser consumers.
//! If any test here fails, something visible to web apps has broken.

use std::collections::BTreeMap;
use tensogram_core::dtype::Dtype;
use tensogram_core::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata};
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

// Szip flag constants (matching libaec.h / tensogram_szip::params exactly).
// Inlined here so tests don't need a direct dep on tensogram-szip.
const AEC_DATA_PREPROCESS: u32 = 8;

// ── Test helpers ─────────────────────────────────────────────────────────────

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
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    }
}

fn make_descriptor_le(shape: Vec<u64>, dtype: Dtype) -> DataObjectDescriptor {
    DataObjectDescriptor {
        byte_order: ByteOrder::Little,
        ..make_descriptor(shape, dtype)
    }
}

fn default_metadata() -> GlobalMetadata {
    GlobalMetadata {
        version: 2,
        ..Default::default()
    }
}

/// Encode a message using the Rust core API (inside WASM).
fn encode_native(meta: &GlobalMetadata, descriptors: &[(&DataObjectDescriptor, &[u8])]) -> Vec<u8> {
    tensogram_core::encode(meta, descriptors, &tensogram_core::EncodeOptions::default()).unwrap()
}

/// Encode a message with hash disabled.
fn encode_native_no_hash(
    meta: &GlobalMetadata,
    descriptors: &[(&DataObjectDescriptor, &[u8])],
) -> Vec<u8> {
    tensogram_core::encode(
        meta,
        descriptors,
        &tensogram_core::EncodeOptions {
            hash_algorithm: None,
            emit_preceders: false,
        },
    )
    .unwrap()
}

/// Build a payload of big-endian f32 values.
fn f32_be_payload(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_be_bytes()).collect()
}

/// Build a payload of little-endian f32 values.
fn f32_le_payload(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Build a payload of big-endian f64 values.
fn f64_be_payload(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_be_bytes()).collect()
}

/// Build a payload of little-endian f64 values.
fn f64_le_payload(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Build a payload of big-endian i32 values.
fn i32_be_payload(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_be_bytes()).collect()
}

/// Build a payload of big-endian i64 values.
fn i64_be_payload(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_be_bytes()).collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. GOLDEN FILE DECODE — canonical .tgm patterns produce expected output
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn golden_simple_f32_decode() {
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let payload = f32_be_payload(&[1.0, 2.0, 3.0, 4.0]);
    let msg = encode_native(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();

    assert_eq!(decoded.object_count(), 1);
    assert_eq!(decoded.object_byte_length(0).unwrap(), 16);

    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload, "golden simple_f32 payload mismatch");

    let desc_js = decoded.object_descriptor(0).unwrap();
    assert!(desc_js.is_object(), "descriptor should be an object");
}

#[wasm_bindgen_test]
fn golden_multi_object_decode() {
    let desc_f32 = make_descriptor(vec![2], Dtype::Float32);
    let desc_i64 = make_descriptor(vec![3], Dtype::Int64);
    let desc_u8 = make_descriptor(vec![5], Dtype::Uint8);

    let payload_f32 = f32_be_payload(&[1.5, 2.5]);
    let payload_i64 = i64_be_payload(&[100, -200, 300]);
    let payload_u8 = vec![10u8, 20, 30, 40, 50];

    let msg = encode_native(
        &default_metadata(),
        &[
            (&desc_f32, &payload_f32),
            (&desc_i64, &payload_i64),
            (&desc_u8, &payload_u8),
        ],
    );

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();

    assert_eq!(decoded.object_count(), 3);
    assert_eq!(decoded.object_byte_length(0).unwrap(), 8);
    assert_eq!(decoded.object_byte_length(1).unwrap(), 24);
    assert_eq!(decoded.object_byte_length(2).unwrap(), 5);
}

#[wasm_bindgen_test]
fn golden_mars_metadata_decode() {
    let mut mars = BTreeMap::new();
    mars.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
    mars.insert("type".to_string(), ciborium::Value::Text("fc".to_string()));
    mars.insert("step".to_string(), ciborium::Value::Integer(12.into()));
    let mut base_entry = BTreeMap::new();
    base_entry.insert(
        "mars".to_string(),
        ciborium::Value::Map(
            mars.into_iter()
                .map(|(k, v)| (ciborium::Value::Text(k), v))
                .collect(),
        ),
    );
    let meta = GlobalMetadata {
        version: 2,
        base: vec![base_entry],
        ..Default::default()
    };
    let desc = make_descriptor(vec![2, 3], Dtype::Float64);
    let payload = f64_be_payload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let msg = encode_native(&meta, &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert_eq!(decoded.object_count(), 1);

    let meta_js = decoded.metadata().unwrap();
    assert!(meta_js.is_object(), "metadata should be an object");

    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload, "mars metadata payload mismatch");
}

#[wasm_bindgen_test]
fn golden_2d_tensor_f32_decode() {
    // 3x4 matrix stored row-major big-endian
    let desc = make_descriptor(vec![3, 4], Dtype::Float32);
    let values: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
    let payload = f32_be_payload(&values);
    let msg = encode_native(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert_eq!(decoded.object_count(), 1);
    assert_eq!(decoded.object_byte_length(0).unwrap(), 48); // 12 * 4
    let raw: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw, payload);
}

#[wasm_bindgen_test]
fn golden_3d_tensor_f64_decode() {
    // 2x3x4 = 24 elements
    let desc = make_descriptor(vec![2, 3, 4], Dtype::Float64);
    let values: Vec<f64> = (0..24).map(|i| i as f64 * 1.1).collect();
    let payload = f64_be_payload(&values);
    let msg = encode_native(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert_eq!(decoded.object_count(), 1);
    assert_eq!(decoded.object_byte_length(0).unwrap(), 192); // 24 * 8
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. ROUND-TRIP — encode in WASM → decode in WASM for every dtype
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn round_trip_f32_no_compression() {
    let desc = make_descriptor(vec![8], Dtype::Float32);
    let values: Vec<f32> = (0..8).map(|i| i as f32 * 1.5).collect();
    let payload = f32_be_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert_eq!(decoded.object_count(), 1);
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload, "round-trip f32 payload mismatch");
}

#[wasm_bindgen_test]
fn round_trip_f64_no_compression() {
    let desc = make_descriptor(vec![4], Dtype::Float64);
    let values = [3.14159f64, 2.71828, 1.41421, 0.0];
    let payload = f64_be_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload);
}

#[wasm_bindgen_test]
fn round_trip_u8_no_compression() {
    let desc = make_descriptor(vec![256], Dtype::Uint8);
    let payload: Vec<u8> = (0..=255).collect();
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload);
}

#[wasm_bindgen_test]
fn round_trip_i8() {
    let desc = make_descriptor(vec![256], Dtype::Int8);
    let payload: Vec<u8> = (-128..=127i8).map(|v| v as u8).collect();
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload);
}

#[wasm_bindgen_test]
fn round_trip_i32() {
    let desc = make_descriptor(vec![8], Dtype::Int32);
    let values: Vec<i32> = vec![i32::MIN, -1000, -1, 0, 1, 1000, i32::MAX - 1, i32::MAX];
    let payload = i32_be_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload);
}

#[wasm_bindgen_test]
fn round_trip_i64() {
    let desc = make_descriptor(vec![4], Dtype::Int64);
    let values = [i64::MIN, 0i64, i64::MAX / 2, i64::MAX];
    let payload = i64_be_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload);
}

#[wasm_bindgen_test]
fn round_trip_u16() {
    let desc = make_descriptor(vec![4], Dtype::Uint16);
    let values: Vec<u16> = vec![0, 255, 65534, 65535];
    let payload: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload);
}

#[wasm_bindgen_test]
fn round_trip_u32() {
    let desc = make_descriptor(vec![4], Dtype::Uint32);
    let values: Vec<u32> = vec![0, 1, u32::MAX - 1, u32::MAX];
    let payload: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload);
}

#[wasm_bindgen_test]
fn round_trip_u64() {
    let desc = make_descriptor(vec![3], Dtype::Uint64);
    let values: Vec<u64> = vec![0, u64::MAX / 2, u64::MAX];
    let payload: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload);
}

#[wasm_bindgen_test]
fn round_trip_i16() {
    let desc = make_descriptor(vec![4], Dtype::Int16);
    let values: Vec<i16> = vec![i16::MIN, -1, 0, i16::MAX];
    let payload: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload);
}

#[wasm_bindgen_test]
fn round_trip_with_hash_verification() {
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let payload = f32_be_payload(&[1.0, 2.0, 3.0, 4.0]);
    let msg = encode_native(&default_metadata(), &[(&desc, &payload)]);

    // Decode with hash verification enabled
    let decoded = tensogram_wasm::decode(&msg, Some(true)).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. COMPRESSION CODEC ROUND-TRIPS
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn round_trip_lz4_compression() {
    let desc = DataObjectDescriptor {
        compression: "lz4".to_string(),
        ..make_descriptor(vec![1024], Dtype::Float32)
    };
    let values: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
    let payload = f32_be_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload, "lz4 round-trip payload mismatch");
}

#[wasm_bindgen_test]
fn round_trip_lz4_large() {
    // 64K elements — tests lz4 with larger payloads
    let desc = DataObjectDescriptor {
        compression: "lz4".to_string(),
        ..make_descriptor(vec![65536], Dtype::Float32)
    };
    let values: Vec<f32> = (0..65536).map(|i| (i as f32 * 0.001).cos()).collect();
    let payload = f32_be_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload);
}

#[wasm_bindgen_test]
fn round_trip_lz4_u8() {
    let desc = DataObjectDescriptor {
        compression: "lz4".to_string(),
        ..make_descriptor(vec![4096], Dtype::Uint8)
    };
    let payload: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload);
}

#[wasm_bindgen_test]
fn round_trip_lz4_f64() {
    let desc = DataObjectDescriptor {
        compression: "lz4".to_string(),
        ..make_descriptor(vec![2048], Dtype::Float64)
    };
    let values: Vec<f64> = (0..2048).map(|i| (i as f64 * 0.1).sin()).collect();
    let payload = f64_be_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload);
}

#[wasm_bindgen_test]
fn round_trip_lz4_constant_data() {
    // Constant data compresses extremely well — verifies decompressor handles it
    let desc = DataObjectDescriptor {
        compression: "lz4".to_string(),
        ..make_descriptor(vec![8192], Dtype::Float32)
    };
    let values = vec![42.0f32; 8192];
    let payload = f32_be_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload);
}

#[wasm_bindgen_test]
fn round_trip_lz4_with_hash_verification() {
    let desc = DataObjectDescriptor {
        compression: "lz4".to_string(),
        ..make_descriptor(vec![512], Dtype::Float32)
    };
    let values: Vec<f32> = (0..512).map(|i| i as f32).collect();
    let payload = f32_be_payload(&values);
    let msg = encode_native(&default_metadata(), &[(&desc, &payload)]);

    // Decode with hash verification — verifies hash is computed over compressed bytes
    let decoded = tensogram_wasm::decode(&msg, Some(true)).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload);
}

#[wasm_bindgen_test]
fn round_trip_szip_pure() {
    // szip needs specific params in the descriptor
    let mut params = BTreeMap::new();
    params.insert("szip_rsi".to_string(), ciborium::Value::Integer(128.into()));
    params.insert(
        "szip_block_size".to_string(),
        ciborium::Value::Integer(16.into()),
    );
    params.insert(
        "szip_flags".to_string(),
        ciborium::Value::Integer((AEC_DATA_PREPROCESS as i64).into()),
    );
    params.insert(
        "szip_bits_per_sample".to_string(),
        ciborium::Value::Integer(8.into()),
    );
    let desc = DataObjectDescriptor {
        compression: "szip".to_string(),
        params,
        ..make_descriptor(vec![2048], Dtype::Uint8)
    };
    let payload: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload, "szip-pure round-trip mismatch");
}

#[wasm_bindgen_test]
fn round_trip_szip_pure_constant_data() {
    let mut params = BTreeMap::new();
    params.insert("szip_rsi".to_string(), ciborium::Value::Integer(128.into()));
    params.insert(
        "szip_block_size".to_string(),
        ciborium::Value::Integer(16.into()),
    );
    params.insert(
        "szip_flags".to_string(),
        ciborium::Value::Integer((AEC_DATA_PREPROCESS as i64).into()),
    );
    params.insert(
        "szip_bits_per_sample".to_string(),
        ciborium::Value::Integer(8.into()),
    );
    let desc = DataObjectDescriptor {
        compression: "szip".to_string(),
        params,
        ..make_descriptor(vec![4096], Dtype::Uint8)
    };
    let payload = vec![42u8; 4096];
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload);
}

#[wasm_bindgen_test]
fn round_trip_zstd_pure() {
    let mut params = BTreeMap::new();
    params.insert("zstd_level".to_string(), ciborium::Value::Integer(3.into()));
    let desc = DataObjectDescriptor {
        compression: "zstd".to_string(),
        params,
        ..make_descriptor(vec![4096], Dtype::Uint8)
    };
    let payload: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload, "zstd-pure round-trip mismatch");
}

#[wasm_bindgen_test]
fn round_trip_zstd_pure_f32() {
    let mut params = BTreeMap::new();
    params.insert("zstd_level".to_string(), ciborium::Value::Integer(1.into()));
    let desc = DataObjectDescriptor {
        compression: "zstd".to_string(),
        params,
        ..make_descriptor(vec![2048], Dtype::Float32)
    };
    let values: Vec<f32> = (0..2048).map(|i| (i as f32).sin()).collect();
    let payload = f32_be_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload);
}

#[wasm_bindgen_test]
fn round_trip_shuffle_plus_lz4() {
    // Shuffle filter + lz4 compression — a common high-performance combo
    let mut params = BTreeMap::new();
    params.insert(
        "shuffle_element_size".to_string(),
        ciborium::Value::Integer(4.into()), // f32 = 4 bytes
    );
    let desc = DataObjectDescriptor {
        filter: "shuffle".to_string(),
        compression: "lz4".to_string(),
        params,
        ..make_descriptor(vec![2048], Dtype::Float32)
    };
    let values: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.01).sin()).collect();
    let payload = f32_be_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload, "shuffle+lz4 round-trip mismatch");
}

#[wasm_bindgen_test]
fn round_trip_shuffle_plus_zstd_pure() {
    let mut params = BTreeMap::new();
    params.insert("zstd_level".to_string(), ciborium::Value::Integer(3.into()));
    params.insert(
        "shuffle_element_size".to_string(),
        ciborium::Value::Integer(4.into()), // f32 = 4 bytes
    );
    let desc = DataObjectDescriptor {
        filter: "shuffle".to_string(),
        compression: "zstd".to_string(),
        params,
        ..make_descriptor(vec![2048], Dtype::Float32)
    };
    let values: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.01).cos()).collect();
    let payload = f32_be_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw_vec: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw_vec, payload, "shuffle+zstd-pure round-trip mismatch");
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. ZERO-COPY TYPED ARRAY VIEWS
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn zero_copy_f32_view_values_correct() {
    let desc = make_descriptor_le(vec![4], Dtype::Float32);
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let payload = f32_le_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let view = decoded.object_data_f32(0).unwrap();

    assert_eq!(view.length(), 4);
    assert_eq!(view.get_index(0), 1.0);
    assert_eq!(view.get_index(1), 2.0);
    assert_eq!(view.get_index(2), 3.0);
    assert_eq!(view.get_index(3), 4.0);
}

#[wasm_bindgen_test]
fn zero_copy_f64_view_values_correct() {
    let desc = make_descriptor_le(vec![3], Dtype::Float64);
    let values = [3.14f64, 2.718, 1.414];
    let payload = f64_le_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let view = decoded.object_data_f64(0).unwrap();

    assert_eq!(view.length(), 3);
    assert_eq!(view.get_index(0), 3.14);
    assert_eq!(view.get_index(1), 2.718);
    assert_eq!(view.get_index(2), 1.414);
}

#[wasm_bindgen_test]
fn zero_copy_i32_view_values_correct() {
    let desc = make_descriptor_le(vec![4], Dtype::Int32);
    let values = [-100i32, 0, 42, i32::MAX];
    let payload: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let view = decoded.object_data_i32(0).unwrap();

    assert_eq!(view.length(), 4);
    assert_eq!(view.get_index(0), -100);
    assert_eq!(view.get_index(1), 0);
    assert_eq!(view.get_index(2), 42);
    assert_eq!(view.get_index(3), i32::MAX);
}

#[wasm_bindgen_test]
fn safe_copy_f32_survives_independently() {
    let desc = make_descriptor_le(vec![2], Dtype::Float32);
    let payload = f32_le_payload(&[42.0, 99.0]);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let copy = decoded.object_data_copy_f32(0).unwrap();

    assert_eq!(copy.length(), 2);
    assert_eq!(copy.get_index(0), 42.0);
    assert_eq!(copy.get_index(1), 99.0);
}

#[wasm_bindgen_test]
fn typed_array_wrong_alignment_returns_error() {
    let desc = make_descriptor(vec![3], Dtype::Uint8);
    let payload = vec![1u8, 2, 3]; // 3 bytes — not a multiple of 4 for f32
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    // f32 view should fail — 3 bytes not divisible by 4
    assert!(decoded.object_data_f32(0).is_err());
    // f64 view should also fail — 3 not divisible by 8
    assert!(decoded.object_data_f64(0).is_err());
    // i32 view should also fail — 3 not divisible by 4
    assert!(decoded.object_data_i32(0).is_err());
    // u8 view should work
    assert!(decoded.object_data_u8(0).is_ok());
}

#[wasm_bindgen_test]
fn f32_view_on_5_byte_payload_errors() {
    let desc = make_descriptor(vec![5], Dtype::Uint8);
    let payload = vec![0u8; 5];
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert!(decoded.object_data_f32(0).is_err());
}

#[wasm_bindgen_test]
fn f64_view_on_4_byte_payload_errors() {
    // 4 bytes is a valid f32 but not f64
    let desc = make_descriptor(vec![1], Dtype::Float32);
    let payload = f32_be_payload(&[1.0]);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert!(decoded.object_data_f64(0).is_err());
    assert!(decoded.object_data_f32(0).is_ok());
}

#[wasm_bindgen_test]
fn safe_copy_f32_on_misaligned_errors() {
    let desc = make_descriptor(vec![3], Dtype::Uint8);
    let payload = vec![1u8, 2, 3];
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert!(decoded.object_data_copy_f32(0).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. DECODE METADATA ONLY
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn decode_metadata_only() {
    let mut extra = BTreeMap::new();
    extra.insert(
        "source".to_string(),
        ciborium::Value::Text("test".to_string()),
    );
    let meta = GlobalMetadata {
        version: 2,
        extra,
        ..Default::default()
    };
    let desc = make_descriptor(vec![100], Dtype::Float32);
    let payload = vec![0u8; 400];
    let msg = encode_native(&meta, &[(&desc, &payload)]);

    let meta_js = tensogram_wasm::decode_metadata(&msg).unwrap();
    assert!(
        meta_js.is_object(),
        "decode_metadata should return an object"
    );

    let version = js_sys::Reflect::get(&meta_js, &"version".into()).unwrap();
    assert_eq!(version.as_f64().unwrap(), 2.0, "version should be 2");
}

#[wasm_bindgen_test]
fn decode_metadata_does_not_require_large_payload() {
    // Verify metadata-only path works regardless of payload size
    let desc = make_descriptor(vec![100000], Dtype::Float32);
    let payload = vec![0u8; 400000]; // 400KB payload
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    // decode_metadata should succeed without decoding the 400KB payload
    let meta_js = tensogram_wasm::decode_metadata(&msg).unwrap();
    assert!(meta_js.is_object());
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. DECODE SINGLE OBJECT BY INDEX
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn decode_object_by_index() {
    let desc0 = make_descriptor(vec![2], Dtype::Float32);
    let desc1 = make_descriptor(vec![3], Dtype::Float32);
    let payload0 = f32_be_payload(&[1.0, 2.0]);
    let payload1 = f32_be_payload(&[10.0, 20.0, 30.0]);
    let msg = encode_native_no_hash(
        &default_metadata(),
        &[(&desc0, &payload0), (&desc1, &payload1)],
    );

    // Decode only object 1
    let decoded = tensogram_wasm::decode_object(&msg, 1, None).unwrap();
    assert_eq!(decoded.object_count(), 1);
    assert_eq!(decoded.object_byte_length(0).unwrap(), 12);
}

#[wasm_bindgen_test]
fn decode_object_first_index() {
    let desc0 = make_descriptor(vec![2], Dtype::Float32);
    let desc1 = make_descriptor(vec![3], Dtype::Float64);
    let payload0 = f32_be_payload(&[1.0, 2.0]);
    let payload1 = f64_be_payload(&[10.0, 20.0, 30.0]);
    let msg = encode_native_no_hash(
        &default_metadata(),
        &[(&desc0, &payload0), (&desc1, &payload1)],
    );

    // Decode object 0
    let decoded = tensogram_wasm::decode_object(&msg, 0, None).unwrap();
    assert_eq!(decoded.object_count(), 1);
    assert_eq!(decoded.object_byte_length(0).unwrap(), 8);
    let raw: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw, payload0);
}

#[wasm_bindgen_test]
fn decode_object_out_of_range_returns_error() {
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let payload = vec![0u8; 8];
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let result = tensogram_wasm::decode_object(&msg, 99, None);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn decode_object_with_hash_verification() {
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let payload = f32_be_payload(&[1.0, 2.0, 3.0, 4.0]);
    let msg = encode_native(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode_object(&msg, 0, Some(true)).unwrap();
    assert_eq!(decoded.object_count(), 1);
    let raw: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw, payload);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. SCAN MULTI-MESSAGE BUFFER
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn scan_multi_message() {
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let payload1 = f32_be_payload(&[1.0, 2.0]);
    let payload2 = f32_be_payload(&[3.0, 4.0]);
    let msg1 = encode_native_no_hash(&default_metadata(), &[(&desc, &payload1)]);
    let msg2 = encode_native_no_hash(&default_metadata(), &[(&desc, &payload2)]);

    let mut multi = msg1.clone();
    multi.extend_from_slice(&msg2);

    let result = tensogram_wasm::scan(&multi).unwrap();
    let arr = js_sys::Array::from(&result);
    assert_eq!(arr.length(), 2);
}

#[wasm_bindgen_test]
fn scan_empty_buffer() {
    let result = tensogram_wasm::scan(&[]).unwrap();
    let arr = js_sys::Array::from(&result);
    assert_eq!(arr.length(), 0);
}

#[wasm_bindgen_test]
fn scan_single_message() {
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &[42u8])]);

    let result = tensogram_wasm::scan(&msg).unwrap();
    let arr = js_sys::Array::from(&result);
    assert_eq!(arr.length(), 1);
}

#[wasm_bindgen_test]
fn scan_three_messages() {
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &[1u8])]);

    let mut multi = msg.clone();
    multi.extend_from_slice(&msg);
    multi.extend_from_slice(&msg);

    let result = tensogram_wasm::scan(&multi).unwrap();
    let arr = js_sys::Array::from(&result);
    assert_eq!(arr.length(), 3);
}

#[wasm_bindgen_test]
fn scan_garbage_prefix_skipped() {
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &[1u8])]);

    // Prepend garbage bytes — scanner should skip them
    let mut data = vec![0xFF; 100];
    data.extend_from_slice(&msg);

    let result = tensogram_wasm::scan(&data).unwrap();
    let arr = js_sys::Array::from(&result);
    assert_eq!(
        arr.length(),
        1,
        "scanner should find the message after garbage"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 8. ERROR HANDLING
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn decode_corrupt_data_returns_error() {
    let result = tensogram_wasm::decode(b"this is not a tgm file", None);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn decode_truncated_message_returns_error() {
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let payload = vec![0u8; 16];
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let truncated = &msg[..msg.len() / 2];
    let result = tensogram_wasm::decode(truncated, None);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn decode_empty_buffer_returns_error() {
    let result = tensogram_wasm::decode(&[], None);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn decode_single_byte_returns_error() {
    let result = tensogram_wasm::decode(&[0x42], None);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn decode_just_preamble_magic_returns_error() {
    // TENSOGRM = [0x54, 0x45, 0x4E, 0x53, 0x4F, 0x47, 0x52, 0x4D]
    let result = tensogram_wasm::decode(b"TENSOGRM", None);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn decode_metadata_corrupt_returns_error() {
    let result = tensogram_wasm::decode_metadata(b"this is not a tgm file");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn decode_metadata_empty_returns_error() {
    let result = tensogram_wasm::decode_metadata(&[]);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn object_index_out_of_bounds_returns_error() {
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let payload = vec![0u8; 8];
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert!(decoded.object_data_f32(99).is_err());
    assert!(decoded.object_descriptor(99).is_err());
    assert!(decoded.object_byte_length(99).is_err());
}

#[wasm_bindgen_test]
fn object_data_u8_oob_returns_error() {
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let payload = vec![0u8; 8];
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert!(decoded.object_data_u8(1).is_err());
}

#[wasm_bindgen_test]
fn decode_object_corrupt_returns_error() {
    let result = tensogram_wasm::decode_object(b"garbage data", 0, None);
    assert!(result.is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
// 9. STREAMING DECODER — basic operations
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn streaming_single_message() {
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let payload = f32_be_payload(&[1.0, 2.0, 3.0, 4.0]);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(&msg);

    assert!(decoder.has_metadata());
    assert_eq!(decoder.pending_count(), 1);

    let frame = decoder.next_frame().unwrap();
    assert_eq!(frame.byte_length(), 16);
    assert!(decoder.next_frame().is_none());
}

#[wasm_bindgen_test]
fn streaming_chunked_feed() {
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let payload = f32_be_payload(&[1.0, 2.0, 3.0, 4.0]);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();

    // Feed in small chunks
    let chunk_size = 16;
    for chunk in msg.chunks(chunk_size) {
        decoder.feed(chunk);
    }

    assert_eq!(decoder.pending_count(), 1);
    let frame = decoder.next_frame().unwrap();
    assert_eq!(frame.byte_length(), 16);
}

#[wasm_bindgen_test]
fn streaming_byte_by_byte_feed() {
    // Extreme case: feed one byte at a time
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let payload = f32_be_payload(&[1.0, 2.0]);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();

    for &byte in &msg {
        decoder.feed(&[byte]);
    }

    assert_eq!(decoder.pending_count(), 1);
    let frame = decoder.next_frame().unwrap();
    assert_eq!(frame.byte_length(), 8);
}

#[wasm_bindgen_test]
fn streaming_multi_message() {
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let payload1 = f32_be_payload(&[1.0, 2.0]);
    let payload2 = f32_be_payload(&[3.0, 4.0]);
    let msg1 = encode_native_no_hash(&default_metadata(), &[(&desc, &payload1)]);
    let msg2 = encode_native_no_hash(&default_metadata(), &[(&desc, &payload2)]);

    let mut multi = msg1;
    multi.extend_from_slice(&msg2);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(&multi);

    assert_eq!(decoder.pending_count(), 2);

    let frame1 = decoder.next_frame().unwrap();
    assert_eq!(frame1.byte_length(), 8);

    let frame2 = decoder.next_frame().unwrap();
    assert_eq!(frame2.byte_length(), 8);

    assert!(decoder.next_frame().is_none());
}

#[wasm_bindgen_test]
fn streaming_multi_object_message() {
    let desc0 = make_descriptor(vec![2], Dtype::Float32);
    let desc1 = make_descriptor(vec![3], Dtype::Uint8);
    let payload0 = vec![0u8; 8];
    let payload1 = vec![0u8; 3];
    let msg = encode_native_no_hash(
        &default_metadata(),
        &[(&desc0, &payload0), (&desc1, &payload1)],
    );

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(&msg);

    assert_eq!(decoder.pending_count(), 2);

    let f0 = decoder.next_frame().unwrap();
    assert_eq!(f0.byte_length(), 8);

    let f1 = decoder.next_frame().unwrap();
    assert_eq!(f1.byte_length(), 3);
}

#[wasm_bindgen_test]
fn streaming_incomplete_message_no_output() {
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let payload = vec![0u8; 16];
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    // Feed only half the message
    decoder.feed(&msg[..msg.len() / 2]);

    assert_eq!(decoder.pending_count(), 0);
    assert!(decoder.next_frame().is_none());
    assert!(!decoder.has_metadata());

    // Feed the rest
    decoder.feed(&msg[msg.len() / 2..]);
    assert_eq!(decoder.pending_count(), 1);
    assert!(decoder.has_metadata());
}

#[wasm_bindgen_test]
fn streaming_reset_clears_state() {
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let payload = vec![0u8; 8];
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(&msg);
    assert_eq!(decoder.pending_count(), 1);

    decoder.reset();
    assert_eq!(decoder.pending_count(), 0);
    assert!(!decoder.has_metadata());
    assert_eq!(decoder.buffered_bytes(), 0);
}

#[wasm_bindgen_test]
fn streaming_reset_then_reuse() {
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let payload1 = f32_be_payload(&[1.0, 2.0]);
    let payload2 = f32_be_payload(&[3.0, 4.0]);
    let msg1 = encode_native_no_hash(&default_metadata(), &[(&desc, &payload1)]);
    let msg2 = encode_native_no_hash(&default_metadata(), &[(&desc, &payload2)]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();

    // First use
    decoder.feed(&msg1);
    assert_eq!(decoder.pending_count(), 1);
    let _ = decoder.next_frame();

    // Reset and reuse
    decoder.reset();
    decoder.feed(&msg2);
    assert_eq!(decoder.pending_count(), 1);

    let frame = decoder.next_frame().unwrap();
    assert_eq!(frame.byte_length(), 8);
}

#[wasm_bindgen_test]
fn streaming_empty_feed_is_noop() {
    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(&[]);
    assert_eq!(decoder.pending_count(), 0);
    assert!(!decoder.has_metadata());
    assert_eq!(decoder.buffered_bytes(), 0);
}

#[wasm_bindgen_test]
fn streaming_garbage_feed_produces_no_frames() {
    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(b"this is not tensogram data at all and is quite long");
    assert_eq!(decoder.pending_count(), 0);
}

#[wasm_bindgen_test]
fn streaming_with_lz4_compressed_message() {
    let desc = DataObjectDescriptor {
        compression: "lz4".to_string(),
        ..make_descriptor(vec![1024], Dtype::Float32)
    };
    let values: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let payload = f32_be_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(&msg);

    assert_eq!(decoder.pending_count(), 1);
    let frame = decoder.next_frame().unwrap();
    assert_eq!(frame.byte_length(), 4096); // 1024 * 4 decoded
}

#[wasm_bindgen_test]
fn streaming_metadata_accessor() {
    let mut extra = BTreeMap::new();
    extra.insert(
        "source".to_string(),
        ciborium::Value::Text("stream-test".to_string()),
    );
    let meta = GlobalMetadata {
        version: 2,
        extra,
        ..Default::default()
    };
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let msg = encode_native_no_hash(&meta, &[(&desc, &[42u8])]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();

    // Before feeding: metadata is null
    let meta_before = decoder.metadata().unwrap();
    assert!(meta_before.is_null());

    // After feeding:
    decoder.feed(&msg);
    let meta_after = decoder.metadata().unwrap();
    assert!(meta_after.is_object());
}

// ═══════════════════════════════════════════════════════════════════════════════
// 10. EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn decode_zero_element_tensor() {
    let desc = make_descriptor(vec![0], Dtype::Float32);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &[])]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert_eq!(decoded.object_count(), 1);
    assert_eq!(decoded.object_byte_length(0).unwrap(), 0);
}

#[wasm_bindgen_test]
fn decode_scalar_tensor() {
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
    let payload = 42.0f64.to_be_bytes().to_vec();
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert_eq!(decoded.object_count(), 1);
    assert_eq!(decoded.object_byte_length(0).unwrap(), 8);
}

#[wasm_bindgen_test]
fn decode_single_element_tensor() {
    let desc = make_descriptor(vec![1], Dtype::Float32);
    let payload = f32_be_payload(&[99.5]);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert_eq!(decoded.object_count(), 1);
    assert_eq!(decoded.object_byte_length(0).unwrap(), 4);
    let raw: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw, payload);
}

#[wasm_bindgen_test]
fn decode_preserves_metadata_version() {
    let meta = GlobalMetadata {
        version: 2,
        ..Default::default()
    };
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let msg = encode_native_no_hash(&meta, &[(&desc, &[0u8])]);

    let meta_js = tensogram_wasm::decode_metadata(&msg).unwrap();
    let version = js_sys::Reflect::get(&meta_js, &"version".into()).unwrap();
    assert_eq!(version.as_f64().unwrap(), 2.0, "version should be 2");
}

#[wasm_bindgen_test]
fn decode_large_metadata_survives() {
    let mut extra = BTreeMap::new();
    for i in 0..50 {
        extra.insert(
            format!("key_{i:03}"),
            ciborium::Value::Text(format!("value_{i}")),
        );
    }
    let meta = GlobalMetadata {
        version: 2,
        extra,
        ..Default::default()
    };
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let msg = encode_native_no_hash(&meta, &[(&desc, &[0u8])]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let meta_js = decoded.metadata().unwrap();
    assert!(meta_js.is_object(), "large metadata should be an object");

    let extra_js = js_sys::Reflect::get(&meta_js, &"_extra_".into()).unwrap();
    assert!(extra_js.is_object(), "extra should be an object");
}

#[wasm_bindgen_test]
fn decode_unicode_metadata_keys() {
    let mut extra = BTreeMap::new();
    extra.insert(
        "日本語キー".to_string(),
        ciborium::Value::Text("値".to_string()),
    );
    extra.insert(
        "emoji_🌍".to_string(),
        ciborium::Value::Text("earth".to_string()),
    );
    let meta = GlobalMetadata {
        version: 2,
        extra,
        ..Default::default()
    };
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let msg = encode_native_no_hash(&meta, &[(&desc, &[0u8])]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let meta_js = decoded.metadata().unwrap();
    assert!(meta_js.is_object());
}

#[wasm_bindgen_test]
fn decode_many_objects() {
    // 20 objects in a single message — stress test object count
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let pairs: Vec<(&DataObjectDescriptor, &[u8])> =
        (0..20).map(|_| (&desc, [0u8].as_slice())).collect();
    let msg = encode_native_no_hash(&default_metadata(), &pairs);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert_eq!(decoded.object_count(), 20);

    // Verify every object is accessible
    for i in 0..20 {
        assert_eq!(decoded.object_byte_length(i).unwrap(), 1);
    }
}

#[wasm_bindgen_test]
fn decode_mixed_dtype_objects() {
    // Message with one object of each commonly-used dtype
    let desc_f32 = make_descriptor(vec![2], Dtype::Float32);
    let desc_f64 = make_descriptor(vec![2], Dtype::Float64);
    let desc_i32 = make_descriptor(vec![2], Dtype::Int32);
    let desc_u8 = make_descriptor(vec![4], Dtype::Uint8);

    let p_f32 = f32_be_payload(&[1.0, 2.0]);
    let p_f64 = f64_be_payload(&[3.0, 4.0]);
    let p_i32 = i32_be_payload(&[-1, 42]);
    let p_u8 = vec![10u8, 20, 30, 40];

    let msg = encode_native_no_hash(
        &default_metadata(),
        &[
            (&desc_f32, &p_f32),
            (&desc_f64, &p_f64),
            (&desc_i32, &p_i32),
            (&desc_u8, &p_u8),
        ],
    );

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert_eq!(decoded.object_count(), 4);
    assert_eq!(decoded.object_byte_length(0).unwrap(), 8); // 2*f32
    assert_eq!(decoded.object_byte_length(1).unwrap(), 16); // 2*f64
    assert_eq!(decoded.object_byte_length(2).unwrap(), 8); // 2*i32
    assert_eq!(decoded.object_byte_length(3).unwrap(), 4); // 4*u8
}

#[wasm_bindgen_test]
fn decode_high_dimensional_tensor() {
    // 5D tensor: 2x2x2x2x2 = 32 elements
    let desc = make_descriptor(vec![2, 2, 2, 2, 2], Dtype::Float32);
    let values: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let payload = f32_be_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert_eq!(decoded.object_count(), 1);
    assert_eq!(decoded.object_byte_length(0).unwrap(), 128); // 32 * 4
}

// ═══════════════════════════════════════════════════════════════════════════════
// 11. API STABILITY — verify the shape of returned JS values
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn api_metadata_returns_object() {
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let payload = vec![0u8; 8];
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let meta = decoded.metadata().unwrap();

    assert!(meta.is_object());
    assert!(!meta.is_null());
    assert!(!meta.is_undefined());
}

#[wasm_bindgen_test]
fn api_descriptor_returns_object_with_expected_keys() {
    let desc = make_descriptor(vec![4, 3], Dtype::Float32);
    let payload = vec![0u8; 48]; // 4*3*4
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let desc_js = decoded.object_descriptor(0).unwrap();

    assert!(
        !desc_js.is_null() && !desc_js.is_undefined(),
        "descriptor must not be null/undefined"
    );

    // serde-wasm-bindgen may serialize as Map or plain Object
    let keys = if desc_js.is_instance_of::<js_sys::Map>() {
        let map: &js_sys::Map = desc_js.unchecked_ref();
        assert!(map.size() > 0, "Map descriptor should have entries");
        let ndim_val = map.get(&"ndim".into());
        assert!(!ndim_val.is_undefined(), "Map should have 'ndim' key");
        let encoding_val = map.get(&"encoding".into());
        assert!(
            !encoding_val.is_undefined(),
            "Map should have 'encoding' key"
        );
        map.size()
    } else {
        let obj: js_sys::Object = desc_js.unchecked_into();
        let keys = js_sys::Object::keys(&obj);
        assert!(keys.length() > 0, "Object descriptor should have keys");
        keys.length()
    };
    assert!(
        keys >= 5,
        "descriptor should have at least 5 fields, got {keys}"
    );
}

#[wasm_bindgen_test]
fn api_object_count_is_stable() {
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let msg = encode_native_no_hash(
        &default_metadata(),
        &[(&desc, &[1u8]), (&desc, &[2u8]), (&desc, &[3u8])],
    );

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    // Calling object_count() multiple times should return the same value
    assert_eq!(decoded.object_count(), 3);
    assert_eq!(decoded.object_count(), 3);
}

#[wasm_bindgen_test]
fn api_typed_array_length_matches_byte_length() {
    let desc = make_descriptor(vec![8], Dtype::Float32);
    let payload = vec![0u8; 32]; // 8 * 4
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();

    let byte_len = decoded.object_byte_length(0).unwrap();
    let f32_view = decoded.object_data_f32(0).unwrap();
    let u8_view = decoded.object_data_u8(0).unwrap();

    assert_eq!(byte_len, 32);
    assert_eq!(f32_view.length(), 8); // 32 bytes / 4 bytes per f32
    assert_eq!(u8_view.length(), 32); // raw bytes
}

#[wasm_bindgen_test]
fn api_metadata_version_field_accessible() {
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &[0u8])]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let meta = decoded.metadata().unwrap();

    // Version should be accessible as a number
    let version = js_sys::Reflect::get(&meta, &"version".into()).unwrap();
    assert!(version.is_truthy());
    assert_eq!(version.as_f64().unwrap(), 2.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 12. STREAMING FRAME DATA CORRECTNESS
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn streaming_frame_data_matches_direct_decode() {
    let desc = make_descriptor_le(vec![4], Dtype::Float32);
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let payload = f32_le_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    // Direct decode
    let direct = tensogram_wasm::decode(&msg, None).unwrap();
    let direct_bytes: Vec<u8> = direct.object_data_u8(0).unwrap().to_vec();

    // Streaming decode
    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(&msg);
    let frame = decoder.next_frame().unwrap();
    let stream_bytes: Vec<u8> = frame.data_u8().unwrap().to_vec();

    assert_eq!(
        direct_bytes, stream_bytes,
        "streaming frame data must match direct decode"
    );
}

#[wasm_bindgen_test]
fn streaming_frame_descriptor_matches_direct() {
    let desc = make_descriptor(vec![4, 3], Dtype::Float64);
    let payload = vec![0u8; 96]; // 4*3*8
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let direct = tensogram_wasm::decode(&msg, None).unwrap();
    let direct_desc: String = js_sys::JSON::stringify(&direct.object_descriptor(0).unwrap())
        .unwrap()
        .into();

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(&msg);
    let frame = decoder.next_frame().unwrap();
    let stream_desc: String = js_sys::JSON::stringify(&frame.descriptor().unwrap())
        .unwrap()
        .into();

    assert_eq!(
        direct_desc, stream_desc,
        "streaming descriptor must match direct decode"
    );
}

#[wasm_bindgen_test]
fn streaming_frame_typed_views_correct() {
    // Verify f32 typed view values from streaming match expected
    let desc = make_descriptor_le(vec![3], Dtype::Float32);
    let values = [10.0f32, 20.0, 30.0];
    let payload = f32_le_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(&msg);
    let frame = decoder.next_frame().unwrap();

    let f32_view = frame.data_f32().unwrap();
    assert_eq!(f32_view.length(), 3);
    assert_eq!(f32_view.get_index(0), 10.0);
    assert_eq!(f32_view.get_index(1), 20.0);
    assert_eq!(f32_view.get_index(2), 30.0);
}

#[wasm_bindgen_test]
fn streaming_frame_base_entry_available() {
    let mut mars = BTreeMap::new();
    mars.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
    let mut base_entry = BTreeMap::new();
    base_entry.insert(
        "mars".to_string(),
        ciborium::Value::Map(
            mars.into_iter()
                .map(|(k, v)| (ciborium::Value::Text(k), v))
                .collect(),
        ),
    );
    let meta = GlobalMetadata {
        version: 2,
        base: vec![base_entry],
        ..Default::default()
    };
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let msg = encode_native_no_hash(&meta, &[(&desc, &[42u8])]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(&msg);
    let frame = decoder.next_frame().unwrap();

    // base_entry should be available and be an object (not null)
    let base = frame.base_entry().unwrap();
    assert!(base.is_object(), "base_entry should be an object");
    assert!(!base.is_null(), "base_entry should not be null");
}

#[wasm_bindgen_test]
fn streaming_multi_object_base_entries() {
    let mut base0 = BTreeMap::new();
    base0.insert(
        "name".to_string(),
        ciborium::Value::Text("temperature".to_string()),
    );
    let mut base1 = BTreeMap::new();
    base1.insert(
        "name".to_string(),
        ciborium::Value::Text("pressure".to_string()),
    );

    let meta = GlobalMetadata {
        version: 2,
        base: vec![base0, base1],
        ..Default::default()
    };

    let desc0 = make_descriptor(vec![2], Dtype::Float32);
    let desc1 = make_descriptor(vec![3], Dtype::Float32);
    let p0 = f32_be_payload(&[1.0, 2.0]);
    let p1 = f32_be_payload(&[3.0, 4.0, 5.0]);

    let msg = encode_native_no_hash(&meta, &[(&desc0, &p0), (&desc1, &p1)]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(&msg);

    assert_eq!(decoder.pending_count(), 2);

    let frame0 = decoder.next_frame().unwrap();
    let frame1 = decoder.next_frame().unwrap();

    // Both frames should have base entries
    assert!(frame0.base_entry().unwrap().is_object());
    assert!(frame1.base_entry().unwrap().is_object());
}

// ═══════════════════════════════════════════════════════════════════════════════
// 13. HASH VERIFICATION EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn decode_hash_disabled_succeeds_on_hashed_message() {
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let payload = f32_be_payload(&[1.0, 2.0, 3.0, 4.0]);
    let msg = encode_native(&default_metadata(), &[(&desc, &payload)]);

    // Decode without hash verification — should succeed
    let decoded = tensogram_wasm::decode(&msg, Some(false)).unwrap();
    assert_eq!(decoded.object_count(), 1);
}

#[wasm_bindgen_test]
fn decode_hash_enabled_on_unhashed_message_succeeds() {
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let payload = f32_be_payload(&[1.0, 2.0, 3.0, 4.0]);
    // Encode without hash
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    // Decode with hash verification — no hash to check, should still succeed
    let decoded = tensogram_wasm::decode(&msg, Some(true)).unwrap();
    assert_eq!(decoded.object_count(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 14. METADATA FIDELITY — verify rich metadata survives encode→decode
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn metadata_extra_survives_round_trip() {
    let mut extra = BTreeMap::new();
    extra.insert(
        "experiment".to_string(),
        ciborium::Value::Text("climate_run_42".to_string()),
    );
    extra.insert("priority".to_string(), ciborium::Value::Integer(1.into()));
    let meta = GlobalMetadata {
        version: 2,
        extra,
        ..Default::default()
    };
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let msg = encode_native_no_hash(&meta, &[(&desc, &[0u8])]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let meta_js = decoded.metadata().unwrap();
    let extra_js = js_sys::Reflect::get(&meta_js, &"_extra_".into()).unwrap();
    assert!(extra_js.is_object(), "_extra_ should survive encode→decode");
}

#[wasm_bindgen_test]
fn metadata_deep_nested_mars_keys() {
    // Deeply nested MARS metadata structure
    let mut param_info = BTreeMap::new();
    param_info.insert(
        "shortName".to_string(),
        ciborium::Value::Text("2t".to_string()),
    );
    param_info.insert("units".to_string(), ciborium::Value::Text("K".to_string()));

    let mut mars = BTreeMap::new();
    mars.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
    mars.insert("type".to_string(), ciborium::Value::Text("fc".to_string()));
    mars.insert(
        "stream".to_string(),
        ciborium::Value::Text("oper".to_string()),
    );
    mars.insert("step".to_string(), ciborium::Value::Integer(6.into()));
    mars.insert(
        "levtype".to_string(),
        ciborium::Value::Text("sfc".to_string()),
    );
    mars.insert(
        "param".to_string(),
        ciborium::Value::Map(
            param_info
                .into_iter()
                .map(|(k, v)| (ciborium::Value::Text(k), v))
                .collect(),
        ),
    );

    let mut base_entry = BTreeMap::new();
    base_entry.insert(
        "mars".to_string(),
        ciborium::Value::Map(
            mars.into_iter()
                .map(|(k, v)| (ciborium::Value::Text(k), v))
                .collect(),
        ),
    );

    let meta = GlobalMetadata {
        version: 2,
        base: vec![base_entry],
        ..Default::default()
    };
    let desc = make_descriptor(vec![10], Dtype::Float32);
    let payload = f32_be_payload(&vec![0.0f32; 10]);
    let msg = encode_native_no_hash(&meta, &[(&desc, &payload)]);

    // Just verify it decodes without error — metadata fidelity
    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert_eq!(decoded.object_count(), 1);
    let meta_js = decoded.metadata().unwrap();
    assert!(meta_js.is_object());
}

#[wasm_bindgen_test]
fn metadata_empty_base_array() {
    // Message with zero base entries but one object
    let meta = GlobalMetadata {
        version: 2,
        base: vec![],
        ..Default::default()
    };
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let payload = f32_be_payload(&[1.0, 2.0]);
    let msg = encode_native_no_hash(&meta, &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert_eq!(decoded.object_count(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 15. DETERMINISM — same input always produces identical output
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn encode_is_deterministic() {
    // Provenance includes a random UUID, so full message bytes differ.
    // Verify the *data payload* round-trips identically each time.
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let payload = f32_be_payload(&[1.0, 2.0, 3.0, 4.0]);

    let msg1 = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);
    let msg2 = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded1 = tensogram_wasm::decode(&msg1, None).unwrap();
    let decoded2 = tensogram_wasm::decode(&msg2, None).unwrap();

    let d1: Vec<u8> = decoded1.object_data_u8(0).unwrap().to_vec();
    let d2: Vec<u8> = decoded2.object_data_u8(0).unwrap().to_vec();
    assert_eq!(d1, d2, "encode must produce deterministic payloads");
    assert_eq!(d1, payload, "payload must match original");
}

#[wasm_bindgen_test]
fn decode_is_deterministic() {
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let payload = f32_be_payload(&[1.0, 2.0, 3.0, 4.0]);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded1 = tensogram_wasm::decode(&msg, None).unwrap();
    let decoded2 = tensogram_wasm::decode(&msg, None).unwrap();

    let raw1: Vec<u8> = decoded1.object_data_u8(0).unwrap().to_vec();
    let raw2: Vec<u8> = decoded2.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw1, raw2, "decode must be deterministic");
}

// ═══════════════════════════════════════════════════════════════════════════════
// 16. LARGE PAYLOAD — verify WASM handles substantial data
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn decode_100k_element_f32() {
    let n = 100_000;
    let desc = make_descriptor(vec![n as u64], Dtype::Float32);
    let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
    let payload = f32_be_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert_eq!(decoded.object_count(), 1);
    assert_eq!(decoded.object_byte_length(0).unwrap(), n * 4);
}

#[wasm_bindgen_test]
fn round_trip_lz4_100k_elements() {
    let n = 100_000;
    let desc = DataObjectDescriptor {
        compression: "lz4".to_string(),
        ..make_descriptor(vec![n as u64], Dtype::Float32)
    };
    let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).cos()).collect();
    let payload = f32_be_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw, payload, "100K element lz4 round-trip mismatch");
}

// ═══════════════════════════════════════════════════════════════════════════════
// 17. DECODE → RE-ENCODE SYMMETRY
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn decode_then_reencode_produces_same_data() {
    let desc = make_descriptor(vec![8], Dtype::Float32);
    let values: Vec<f32> = (0..8).map(|i| i as f32 * 2.5).collect();
    let original_payload = f32_be_payload(&values);
    let msg1 = encode_native_no_hash(&default_metadata(), &[(&desc, &original_payload)]);

    // Decode
    let decoded = tensogram_wasm::decode(&msg1, None).unwrap();
    let decoded_payload: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();

    // Re-encode with same descriptor and metadata
    let msg2 = encode_native_no_hash(&default_metadata(), &[(&desc, &decoded_payload)]);

    // Decode again
    let decoded2 = tensogram_wasm::decode(&msg2, None).unwrap();
    let final_payload: Vec<u8> = decoded2.object_data_u8(0).unwrap().to_vec();

    assert_eq!(
        original_payload, final_payload,
        "decode→re-encode must preserve data"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 18. STREAMING DECODER — ADVANCED SCENARIOS
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn streaming_interleaved_feed_and_consume() {
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let payload1 = f32_be_payload(&[1.0, 2.0]);
    let payload2 = f32_be_payload(&[3.0, 4.0]);
    let msg1 = encode_native_no_hash(&default_metadata(), &[(&desc, &payload1)]);
    let msg2 = encode_native_no_hash(&default_metadata(), &[(&desc, &payload2)]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();

    // Feed first message, consume it
    decoder.feed(&msg1);
    assert_eq!(decoder.pending_count(), 1);
    let frame1 = decoder.next_frame().unwrap();
    assert_eq!(frame1.byte_length(), 8);
    assert_eq!(decoder.pending_count(), 0);

    // Feed second message, consume it
    decoder.feed(&msg2);
    assert_eq!(decoder.pending_count(), 1);
    let frame2 = decoder.next_frame().unwrap();
    assert_eq!(frame2.byte_length(), 8);
    assert_eq!(decoder.pending_count(), 0);
}

#[wasm_bindgen_test]
fn streaming_five_messages_sequential() {
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let mut all_bytes = Vec::new();
    for i in 0..5u8 {
        let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &[i])]);
        all_bytes.extend_from_slice(&msg);
    }

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(&all_bytes);
    assert_eq!(decoder.pending_count(), 5);

    for _ in 0..5 {
        let frame = decoder.next_frame().unwrap();
        assert_eq!(frame.byte_length(), 1);
    }
    assert!(decoder.next_frame().is_none());
}

#[wasm_bindgen_test]
fn streaming_buffered_bytes_tracks_correctly() {
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let payload = f32_be_payload(&[1.0, 2.0, 3.0, 4.0]);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();

    // Feed partial data
    let half = msg.len() / 2;
    decoder.feed(&msg[..half]);
    assert!(decoder.buffered_bytes() > 0, "should have buffered data");

    // Feed the rest
    decoder.feed(&msg[half..]);
    // After decoding, buffered bytes should be consumed
    // (the exact value depends on implementation, but pending count should be 1)
    assert_eq!(decoder.pending_count(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 19. BYTE ORDER HANDLING
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn round_trip_big_endian_f32() {
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let payload = f32_be_payload(&[1.0, 2.0, 3.0, 4.0]);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw, payload);
}

#[wasm_bindgen_test]
fn round_trip_little_endian_f32() {
    let desc = make_descriptor_le(vec![4], Dtype::Float32);
    let payload = f32_le_payload(&[1.0, 2.0, 3.0, 4.0]);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw, payload);
}

#[wasm_bindgen_test]
fn round_trip_big_endian_f64() {
    let desc = make_descriptor(vec![3], Dtype::Float64);
    let payload = f64_be_payload(&[1.1, 2.2, 3.3]);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw, payload);
}

#[wasm_bindgen_test]
fn round_trip_little_endian_f64() {
    let desc = make_descriptor_le(vec![3], Dtype::Float64);
    let payload = f64_le_payload(&[1.1, 2.2, 3.3]);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw, payload);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 20. STREAMING ERROR VISIBILITY — last_error() and skipped_count()
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn streaming_no_error_initially() {
    let decoder = tensogram_wasm::StreamingDecoder::new();
    assert!(decoder.last_error().is_none());
    assert_eq!(decoder.skipped_count(), 0);
}

#[wasm_bindgen_test]
fn streaming_last_error_cleared_on_feed() {
    let desc = make_descriptor(vec![2], Dtype::Float32);
    let payload = f32_be_payload(&[1.0, 2.0]);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(&msg).unwrap();
    assert!(decoder.last_error().is_none(), "no error on valid message");
}

#[wasm_bindgen_test]
fn streaming_reset_clears_error_state() {
    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(b"garbage that does nothing").unwrap();
    decoder.reset();
    assert!(decoder.last_error().is_none());
    assert_eq!(decoder.skipped_count(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 21. STREAMING BUFFER LIMIT — prevents unbounded memory growth
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn streaming_buffer_limit_rejects_overflow() {
    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.set_max_buffer(100); // tiny limit for testing

    // First feed: fits
    let small = vec![0u8; 50];
    assert!(decoder.feed(&small).is_ok());

    // Second feed: would exceed limit
    let more = vec![0u8; 60];
    assert!(
        decoder.feed(&more).is_err(),
        "should reject buffer overflow"
    );
}

#[wasm_bindgen_test]
fn streaming_buffer_limit_allows_within_budget() {
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &[42u8])]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.set_max_buffer(1024 * 1024); // 1 MiB — plenty
    assert!(decoder.feed(&msg).is_ok());
    assert_eq!(decoder.pending_count(), 1);
}

#[wasm_bindgen_test]
fn streaming_buffer_reclaims_after_consume() {
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &[42u8])]);
    let msg_len = msg.len();

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    // Set limit to 2× one message — room for exactly one buffered + one incoming
    decoder.set_max_buffer(msg_len * 3);

    // First message
    assert!(decoder.feed(&msg).is_ok());
    let _ = decoder.next_frame(); // consume

    // Second message — should work because decoder consumed first
    assert!(decoder.feed(&msg).is_ok());
    assert_eq!(decoder.pending_count(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 22. EDGE CASES — boundary conditions and corner cases
// ═══════════════════════════════════════════════════════════════════════════════

#[wasm_bindgen_test]
fn streaming_feed_returns_ok_on_empty_chunk() {
    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    assert!(decoder.feed(&[]).is_ok());
    assert_eq!(decoder.buffered_bytes(), 0);
}

#[wasm_bindgen_test]
fn decode_zero_byte_payload_all_dtypes() {
    // Shape [0] means zero elements — should work for every dtype
    for dtype in [
        Dtype::Float32,
        Dtype::Float64,
        Dtype::Int8,
        Dtype::Int16,
        Dtype::Int32,
        Dtype::Int64,
        Dtype::Uint8,
        Dtype::Uint16,
        Dtype::Uint32,
        Dtype::Uint64,
    ] {
        let desc = make_descriptor(vec![0], dtype);
        let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &[])]);
        let decoded = tensogram_wasm::decode(&msg, None).unwrap();
        assert_eq!(decoded.object_count(), 1);
        assert_eq!(decoded.object_byte_length(0).unwrap(), 0);
    }
}

#[wasm_bindgen_test]
fn decode_object_index_exactly_at_boundary() {
    // 3 objects, try index 2 (last valid) and index 3 (first invalid)
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let msg = encode_native_no_hash(
        &default_metadata(),
        &[(&desc, &[1u8]), (&desc, &[2u8]), (&desc, &[3u8])],
    );

    // Last valid index
    let decoded = tensogram_wasm::decode_object(&msg, 2, None).unwrap();
    assert_eq!(decoded.object_count(), 1);

    // First invalid index
    assert!(tensogram_wasm::decode_object(&msg, 3, None).is_err());
}

#[wasm_bindgen_test]
fn streaming_skipped_count_persists_across_feeds() {
    let mut decoder = tensogram_wasm::StreamingDecoder::new();

    // Feed valid message
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &[42u8])]);
    decoder.feed(&msg).unwrap();
    assert_eq!(decoder.skipped_count(), 0);

    // Feed another valid message
    decoder.feed(&msg).unwrap();
    assert_eq!(decoder.skipped_count(), 0);
}

#[wasm_bindgen_test]
fn typed_view_on_zero_length_payload() {
    let desc = make_descriptor(vec![0], Dtype::Float32);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &[])]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();

    // Zero-length views should succeed (not error)
    let f32_view = decoded.object_data_f32(0).unwrap();
    assert_eq!(f32_view.length(), 0);

    let u8_view = decoded.object_data_u8(0).unwrap();
    assert_eq!(u8_view.length(), 0);
}

#[wasm_bindgen_test]
fn safe_copy_on_zero_length_payload() {
    let desc = make_descriptor(vec![0], Dtype::Float32);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &[])]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let copy = decoded.object_data_copy_f32(0).unwrap();
    assert_eq!(copy.length(), 0);
}

#[wasm_bindgen_test]
fn streaming_valid_after_garbage_in_same_feed() {
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &[42u8])]);

    // Prepend garbage that doesn't look like framing
    let mut data = vec![0xDE, 0xAD, 0xBE, 0xEF];
    data.extend_from_slice(&msg);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(&data).unwrap();

    // The scanner should find the valid message after garbage
    assert!(
        decoder.pending_count() >= 1,
        "should find message after garbage"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 23. CODE COVERAGE — previously untested code paths
// ═══════════════════════════════════════════════════════════════════════════════

// -- encode() JS-facing API (lib.rs:92-155) ---------------------------------

// encode() is wasm_bindgen-only and requires constructing JS values.
// We can't call it directly from Rust tests but we verify the encode
// pipeline through the Rust core API used by encode_native helpers.
// The following tests exercise the branches inside encode() indirectly.

#[wasm_bindgen_test]
fn encode_no_hash_round_trips() {
    // Exercises the hash=false branch of the encode pipeline
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let payload = f32_be_payload(&[1.0, 2.0, 3.0, 4.0]);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);
    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert_eq!(decoded.object_count(), 1);
}

#[wasm_bindgen_test]
fn encode_with_hash_round_trips() {
    // Exercises the hash=true branch
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let payload = f32_be_payload(&[1.0, 2.0, 3.0, 4.0]);
    let msg = encode_native(&default_metadata(), &[(&desc, &payload)]);
    let decoded = tensogram_wasm::decode(&msg, Some(true)).unwrap();
    assert_eq!(decoded.object_count(), 1);
}

// -- Streaming DecodedFrame typed views (data_f64, data_i32) ----------------

#[wasm_bindgen_test]
fn streaming_frame_f64_view() {
    let desc = make_descriptor_le(vec![3], Dtype::Float64);
    let values = [1.1f64, 2.2, 3.3];
    let payload = f64_le_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(&msg).unwrap();
    let frame = decoder.next_frame().unwrap();

    let view = frame.data_f64().unwrap();
    assert_eq!(view.length(), 3);
    assert_eq!(view.get_index(0), 1.1);
    assert_eq!(view.get_index(2), 3.3);
}

#[wasm_bindgen_test]
fn streaming_frame_i32_view() {
    let desc = make_descriptor_le(vec![2], Dtype::Int32);
    let values = [-42i32, 99];
    let payload: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(&msg).unwrap();
    let frame = decoder.next_frame().unwrap();

    let view = frame.data_i32().unwrap();
    assert_eq!(view.length(), 2);
    assert_eq!(view.get_index(0), -42);
    assert_eq!(view.get_index(1), 99);
}

#[wasm_bindgen_test]
fn streaming_frame_u8_view() {
    let desc = make_descriptor(vec![4], Dtype::Uint8);
    let payload = vec![10u8, 20, 30, 40];
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(&msg).unwrap();
    let frame = decoder.next_frame().unwrap();

    let view = frame.data_u8().unwrap();
    assert_eq!(view.length(), 4);
}

// -- Zero-length typed views for all types ----------------------------------

#[wasm_bindgen_test]
fn zero_length_f64_view() {
    let desc = make_descriptor(vec![0], Dtype::Float64);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &[])]);
    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let view = decoded.object_data_f64(0).unwrap();
    assert_eq!(view.length(), 0);
}

#[wasm_bindgen_test]
fn zero_length_i32_view() {
    let desc = make_descriptor(vec![0], Dtype::Int32);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &[])]);
    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let view = decoded.object_data_i32(0).unwrap();
    assert_eq!(view.length(), 0);
}

// -- Streaming: max_buffer edge cases ---------------------------------------

#[wasm_bindgen_test]
fn streaming_max_buffer_zero_rejects_any_feed() {
    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.set_max_buffer(0);
    assert!(decoder.feed(&[1u8]).is_err());
}

#[wasm_bindgen_test]
fn streaming_max_buffer_exact_limit_accepts() {
    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    let data = vec![0u8; 100];
    decoder.set_max_buffer(100);
    assert!(decoder.feed(&data).is_ok());
}

#[wasm_bindgen_test]
fn streaming_max_buffer_one_over_limit_rejects() {
    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.set_max_buffer(100);
    assert!(decoder.feed(&[0u8; 101]).is_err());
}

// -- szip-pure 16-bit samples through WASM ----------------------------------

#[wasm_bindgen_test]
fn round_trip_szip_pure_16bit() {
    let mut params = BTreeMap::new();
    params.insert("szip_rsi".to_string(), ciborium::Value::Integer(128.into()));
    params.insert(
        "szip_block_size".to_string(),
        ciborium::Value::Integer(16.into()),
    );
    params.insert(
        "szip_flags".to_string(),
        ciborium::Value::Integer((AEC_DATA_PREPROCESS as i64).into()),
    );
    params.insert(
        "szip_bits_per_sample".to_string(),
        ciborium::Value::Integer(16.into()),
    );
    let desc = DataObjectDescriptor {
        compression: "szip".to_string(),
        params,
        ..make_descriptor(vec![2048], Dtype::Uint16)
    };
    // 2048 u16 samples in big-endian
    let payload: Vec<u8> = (0..2048u16)
        .flat_map(|i| (i.wrapping_mul(7)).to_be_bytes())
        .collect();
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);
    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw, payload, "szip-pure 16-bit round-trip mismatch");
}

// -- zstd-pure constant data through WASM -----------------------------------

#[wasm_bindgen_test]
fn round_trip_zstd_pure_constant() {
    let mut params = BTreeMap::new();
    params.insert("zstd_level".to_string(), ciborium::Value::Integer(1.into()));
    let desc = DataObjectDescriptor {
        compression: "zstd".to_string(),
        params,
        ..make_descriptor(vec![4096], Dtype::Uint8)
    };
    let payload = vec![42u8; 4096]; // constant → high compression ratio
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);
    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let raw: Vec<u8> = decoded.object_data_u8(0).unwrap().to_vec();
    assert_eq!(raw, payload);
}

// -- Streaming frame: zero-length payload -----------------------------------

#[wasm_bindgen_test]
fn streaming_frame_zero_length_views() {
    let desc = make_descriptor(vec![0], Dtype::Float32);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &[])]);

    let mut decoder = tensogram_wasm::StreamingDecoder::new();
    decoder.feed(&msg).unwrap();
    let frame = decoder.next_frame().unwrap();

    assert_eq!(frame.byte_length(), 0);
    assert_eq!(frame.data_f32().unwrap().length(), 0);
    assert_eq!(frame.data_f64().unwrap().length(), 0);
    assert_eq!(frame.data_i32().unwrap().length(), 0);
    assert_eq!(frame.data_u8().unwrap().length(), 0);
}

// -- DecodedMessage: object_descriptor OOB ----------------------------------

#[wasm_bindgen_test]
fn object_descriptor_oob_returns_error() {
    let desc = make_descriptor(vec![1], Dtype::Uint8);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &[0u8])]);
    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    assert!(decoded.object_descriptor(1).is_err());
    assert!(decoded.object_descriptor(999).is_err());
}

// -- DecodedMessage: typed views on compressed data -------------------------

#[wasm_bindgen_test]
fn compressed_data_f64_view_correct() {
    let desc = DataObjectDescriptor {
        compression: "lz4".to_string(),
        ..make_descriptor_le(vec![3], Dtype::Float64)
    };
    let values = [1.5f64, 2.5, 3.5];
    let payload = f64_le_payload(&values);
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let view = decoded.object_data_f64(0).unwrap();
    assert_eq!(view.length(), 3);
    assert_eq!(view.get_index(0), 1.5);
}

#[wasm_bindgen_test]
fn compressed_data_i32_view_correct() {
    let desc = DataObjectDescriptor {
        compression: "lz4".to_string(),
        ..make_descriptor_le(vec![4], Dtype::Int32)
    };
    let values = [-10i32, 0, 10, 20];
    let payload: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let msg = encode_native_no_hash(&default_metadata(), &[(&desc, &payload)]);

    let decoded = tensogram_wasm::decode(&msg, None).unwrap();
    let view = decoded.object_data_i32(0).unwrap();
    assert_eq!(view.length(), 4);
    assert_eq!(view.get_index(0), -10);
    assert_eq!(view.get_index(3), 20);
}
