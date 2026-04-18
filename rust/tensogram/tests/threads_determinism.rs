// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! End-to-end determinism tests for the multi-threaded coding pipeline
//! (introduced in v0.13.0).
//!
//! The core contract is:
//!
//! 1. **Transparent codecs** (encoding = simple_packing, filter = shuffle,
//!    or compression ∈ {none, lz4, szip, zfp, sz3}):
//!    `encode(threads=T)` bytes are identical across `T ∈ {0, 1, 2, 4, 8, 16}`.
//!
//! 2. **Opaque codecs** (compression ∈ {blosc2, zstd with `nb_workers>0`}):
//!    compressed bytes MAY differ across thread counts (the codec writes
//!    blocks in completion order) but the decoded values MUST round-trip
//!    losslessly regardless of how they were encoded.
//!
//! 3. **Golden invariant**: `threads = 0` (the default) must produce
//!    byte-for-byte the same output as the pre-0.13.0 sequential path.
//!    Golden `.tgm` files continue to validate against the new code.

use std::collections::BTreeMap;

use tensogram::{
    ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions, GlobalMetadata, decode,
    encode,
};

/// Extract just the encoded payload bytes of each object from a
/// message buffer.  Equality of this vector across thread counts is
/// the "transparent codec byte-identity" contract — it deliberately
/// ignores the top-level `_reserved_` provenance fields (uuid, time,
/// encoder version) which change on every encode call.
fn encoded_payloads(buf: &[u8]) -> Vec<Vec<u8>> {
    let msg = tensogram::framing::decode_message(buf).expect("decode_message");
    msg.objects
        .iter()
        .map(|(_, payload, _)| payload.to_vec())
        .collect()
}

fn make_descriptor(shape: Vec<u64>, dtype: Dtype) -> DataObjectDescriptor {
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
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    }
}

fn large_float_bytes(n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| (250.0f64 + (i as f64).sin() * 30.0).to_ne_bytes())
        .collect()
}

/// Thread counts we sweep in every determinism test.
const THREAD_COUNTS: &[u32] = &[0, 1, 2, 4, 8, 16];

// ── Transparent codec byte-identity ────────────────────────────────────

#[test]
fn encode_no_encoding_no_compression_byte_identical() {
    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![200_000], Dtype::Float64);
    let data = large_float_bytes(200_000);

    let make_options = |t: u32| EncodeOptions {
        threads: t,
        parallel_threshold_bytes: Some(0), // force parallel path above t>=2
        ..Default::default()
    };
    let baseline = encoded_payloads(&encode(&meta, &[(&desc, &data)], &make_options(0)).unwrap());
    for &t in THREAD_COUNTS {
        let out = encoded_payloads(&encode(&meta, &[(&desc, &data)], &make_options(t)).unwrap());
        assert_eq!(
            baseline, out,
            "encode threads={t} must produce byte-identical payload for transparent pipeline"
        );
    }
}

#[cfg(feature = "lz4")]
#[test]
fn encode_lz4_byte_identical() {
    let meta = GlobalMetadata::default();
    let mut desc = make_descriptor(vec![200_000], Dtype::Float32);
    desc.compression = "lz4".to_string();
    let data: Vec<u8> = (0..200_000)
        .flat_map(|i| (i as f32).sin().to_ne_bytes())
        .collect();

    let make_options = |t: u32| EncodeOptions {
        threads: t,
        parallel_threshold_bytes: Some(0),
        ..Default::default()
    };
    let baseline = encoded_payloads(&encode(&meta, &[(&desc, &data)], &make_options(0)).unwrap());
    for &t in THREAD_COUNTS {
        let out = encoded_payloads(&encode(&meta, &[(&desc, &data)], &make_options(t)).unwrap());
        assert_eq!(baseline, out, "lz4 threads={t} must be byte-identical");
    }
}

#[cfg(any(feature = "szip", feature = "szip-pure"))]
#[test]
fn encode_simple_packing_plus_szip_byte_identical() {
    use ciborium::Value;

    let meta = GlobalMetadata::default();
    let values: Vec<f64> = (0..200_000)
        .map(|i| 250.0 + (i as f64).sin() * 30.0)
        .collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let mut desc = make_descriptor(vec![values.len() as u64], Dtype::Float64);
    let params = tensogram_encodings::simple_packing::compute_params(&values, 24, 0).unwrap();
    desc.encoding = "simple_packing".to_string();
    desc.params.insert(
        "reference_value".to_string(),
        Value::Float(params.reference_value),
    );
    desc.params.insert(
        "binary_scale_factor".to_string(),
        Value::Integer((i64::from(params.binary_scale_factor)).into()),
    );
    desc.params.insert(
        "decimal_scale_factor".to_string(),
        Value::Integer((i64::from(params.decimal_scale_factor)).into()),
    );
    desc.params.insert(
        "bits_per_value".to_string(),
        Value::Integer((i64::from(params.bits_per_value)).into()),
    );
    desc.compression = "szip".to_string();
    desc.params
        .insert("szip_rsi".to_string(), Value::Integer(128.into()));
    desc.params
        .insert("szip_block_size".to_string(), Value::Integer(16.into()));
    desc.params
        .insert("szip_flags".to_string(), Value::Integer(8.into()));

    let make_options = |t: u32| EncodeOptions {
        threads: t,
        parallel_threshold_bytes: Some(0),
        ..Default::default()
    };
    let baseline = encoded_payloads(&encode(&meta, &[(&desc, &data)], &make_options(0)).unwrap());
    for &t in THREAD_COUNTS {
        let out = encoded_payloads(&encode(&meta, &[(&desc, &data)], &make_options(t)).unwrap());
        assert_eq!(
            baseline, out,
            "simple_packing + szip threads={t} must be byte-identical"
        );
    }
}

// ── Opaque codec round-trip invariance ─────────────────────────────────

#[cfg(feature = "blosc2")]
#[test]
fn encode_blosc2_round_trip_lossless_across_threads() {
    use ciborium::Value;

    let meta = GlobalMetadata::default();
    let mut desc = make_descriptor(vec![100_000], Dtype::Float64);
    desc.compression = "blosc2".to_string();
    desc.params
        .insert("blosc2_clevel".to_string(), Value::Integer(5.into()));
    desc.params
        .insert("blosc2_codec".to_string(), Value::Text("lz4".to_string()));

    let data = large_float_bytes(100_000);

    let make_options = |t: u32| EncodeOptions {
        threads: t,
        parallel_threshold_bytes: Some(0),
        ..Default::default()
    };
    for &t in THREAD_COUNTS {
        let out = encode(&meta, &[(&desc, &data)], &make_options(t)).unwrap();
        // Decode with threads=0 to prove cross-thread-count decode works.
        let (_meta, objects) = decode(&out, &DecodeOptions::default()).unwrap();
        assert_eq!(objects.len(), 1);
        assert_eq!(
            objects[0].1, data,
            "blosc2 threads={t} round-trip must be lossless"
        );
    }
}

// ── Multi-object axis-A behaviour ──────────────────────────────────────

#[test]
fn encode_multi_object_no_compression_order_preserved() {
    // 16 objects, no codec (transparent) — exercise axis A.  Order must
    // be preserved even though workers complete in arbitrary order.
    let meta = GlobalMetadata::default();
    let mut descriptors_owned = Vec::new();
    let mut data_owned = Vec::new();
    for i in 0..16u32 {
        descriptors_owned.push(make_descriptor(vec![50_000], Dtype::Uint32));
        let bytes: Vec<u8> = (0..50_000)
            .flat_map(|j| (i * 1_000_000 + j).to_ne_bytes())
            .collect();
        data_owned.push(bytes);
    }
    let pairs: Vec<(&DataObjectDescriptor, &[u8])> = descriptors_owned
        .iter()
        .zip(data_owned.iter())
        .map(|(d, v)| (d, v.as_slice()))
        .collect();

    let make_options = |t: u32| EncodeOptions {
        threads: t,
        parallel_threshold_bytes: Some(0),
        ..Default::default()
    };
    let baseline = encoded_payloads(&encode(&meta, &pairs, &make_options(0)).unwrap());
    for &t in THREAD_COUNTS {
        let out = encoded_payloads(&encode(&meta, &pairs, &make_options(t)).unwrap());
        assert_eq!(
            baseline, out,
            "multi-object axis-A threads={t} must preserve input order byte-identically"
        );

        let bytes = encode(&meta, &pairs, &make_options(t)).unwrap();
        let (_meta, decoded) = decode(&bytes, &DecodeOptions::default()).unwrap();
        assert_eq!(decoded.len(), 16);
        for (i, (_d, data)) in decoded.iter().enumerate() {
            assert_eq!(*data, data_owned[i], "object {i} data mismatch");
        }
    }
}

// ── Threshold behaviour ────────────────────────────────────────────────

#[test]
fn threads_ignored_below_threshold() {
    // 1 KiB payload — way below the default 64 KiB threshold.  Every
    // thread count must produce byte-identical payload because the
    // parallel path is never taken.
    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![128], Dtype::Float64);
    let data = large_float_bytes(128);

    let make_options = |t: u32| EncodeOptions {
        threads: t,
        parallel_threshold_bytes: None,
        ..Default::default()
    };
    let baseline = encoded_payloads(&encode(&meta, &[(&desc, &data)], &make_options(0)).unwrap());
    for &t in THREAD_COUNTS {
        let out = encoded_payloads(&encode(&meta, &[(&desc, &data)], &make_options(t)).unwrap());
        assert_eq!(baseline, out, "tiny payload threads={t} must be identical");
    }
}

// ── Decode determinism ─────────────────────────────────────────────────

#[test]
fn decode_threads_byte_identical_transparent() {
    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![200_000], Dtype::Float64);
    let data = large_float_bytes(200_000);
    let msg = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    let baseline_opts = DecodeOptions::default();
    let (_meta0, objects0) = decode(&msg, &baseline_opts).unwrap();
    for &t in THREAD_COUNTS {
        let opts = DecodeOptions {
            threads: t,
            parallel_threshold_bytes: Some(0),
            ..Default::default()
        };
        let (_meta, objects) = decode(&msg, &opts).unwrap();
        assert_eq!(objects.len(), objects0.len());
        assert_eq!(
            objects[0].1, objects0[0].1,
            "decode threads={t} must match threads=0 byte-identically"
        );
    }
}

// ── Edge cases ─────────────────────────────────────────────────────────

/// A message with no data objects is legal.  `threads=N` must not trip
/// any of the axis-A / axis-B dispatch logic on an empty slice.
#[test]
fn encode_zero_objects_any_threads_ok() {
    let meta = GlobalMetadata::default();
    for &t in &[0u32, 1, 4, 8] {
        let opts = EncodeOptions {
            threads: t,
            parallel_threshold_bytes: Some(0),
            ..Default::default()
        };
        let msg = encode(&meta, &[], &opts).expect("encode with 0 objects");
        // Round-trip: decode must also accept an empty-object message.
        let dec_opts = DecodeOptions {
            threads: t,
            parallel_threshold_bytes: Some(0),
            ..Default::default()
        };
        let (_meta, objects) = decode(&msg, &dec_opts).expect("decode empty");
        assert_eq!(objects.len(), 0);
    }
}

/// A zero-length payload on a legal shape must not crash the parallel
/// dispatcher.  `shape = [0]` is a valid empty tensor.
#[test]
fn encode_zero_length_payload_any_threads_ok() {
    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![0], Dtype::Float64);
    let data: Vec<u8> = vec![];
    for &t in &[0u32, 1, 4, 8] {
        let opts = EncodeOptions {
            threads: t,
            parallel_threshold_bytes: Some(0),
            ..Default::default()
        };
        let msg = encode(&meta, &[(&desc, &data)], &opts).expect("encode zero-length");
        let dec_opts = DecodeOptions {
            threads: t,
            parallel_threshold_bytes: Some(0),
            ..Default::default()
        };
        let (_meta, objects) = decode(&msg, &dec_opts).expect("decode zero-length");
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].1.len(), 0);
    }
}

/// `decode_range` with an empty `ranges` slice is a no-op: must return
/// `Ok(vec![])` regardless of `threads`.
#[test]
fn decode_range_empty_slice_any_threads_ok() {
    use tensogram::decode_range;

    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![1024], Dtype::Float64);
    let data = large_float_bytes(1024);
    let msg = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    for &t in &[0u32, 1, 4, 8] {
        let opts = DecodeOptions {
            threads: t,
            parallel_threshold_bytes: Some(0),
            ..Default::default()
        };
        let (_desc, parts) = decode_range(&msg, 0, &[], &opts).expect("empty ranges");
        assert!(parts.is_empty(), "threads={t} expected no parts");
    }
}

/// When `threads > 1` is requested but no axis has work to distribute
/// (e.g. single tiny object), the dispatcher must still produce correct
/// output and must not construct a pool.  This is harder to assert
/// directly — instead we verify that the output is byte-identical to
/// the sequential path.
#[test]
fn single_tiny_object_threads_ignored() {
    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();

    let baseline =
        encoded_payloads(&encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap());
    for &t in &[1u32, 4, 16] {
        let opts = EncodeOptions {
            threads: t,
            parallel_threshold_bytes: None, // default 64 KiB — well above 16 bytes
            ..Default::default()
        };
        let got = encoded_payloads(&encode(&meta, &[(&desc, &data)], &opts).unwrap());
        assert_eq!(
            baseline, got,
            "single tiny object threads={t} must match sequential"
        );
    }
}
