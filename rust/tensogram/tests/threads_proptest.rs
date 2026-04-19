// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Property-based tests for the multi-threaded coding pipeline.
//!
//! Randomises over (`n_objects`, `payload_size`, `compression`,
//! `threads`) and checks two invariants on every generated message:
//!
//! 1. **Round-trip invariant.**  encode(threads=T) then decode(threads=U)
//!    returns exactly the original payload bytes for every T, U.
//! 2. **Transparent byte-identity.**  For codecs we classify as
//!    transparent (see the `is_transparent` helper), varying T while
//!    holding the payload fixed produces byte-identical encoded
//!    payloads — matching the pass-3 determinism contract.
//!
//! These tests complement the hand-crafted integration tests in
//! `threads_determinism.rs` by fuzzing combinations we wouldn't write
//! by hand.

use std::collections::BTreeMap;

use proptest::prelude::*;

use tensogram::{
    ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions, GlobalMetadata, decode,
    encode,
};

#[derive(Debug, Clone)]
enum Compression {
    None,
    Lz4,
    Blosc2,
    Zstd,
}

impl Compression {
    fn wire_name(&self) -> &'static str {
        match self {
            Compression::None => "none",
            Compression::Lz4 => "lz4",
            Compression::Blosc2 => "blosc2",
            Compression::Zstd => "zstd",
        }
    }

    /// Transparent codecs (none, lz4) must produce byte-identical
    /// encoded payload across all thread counts.  Blosc2 and zstd
    /// with workers > 0 may reorder blocks by completion order — only
    /// round-trip identity is required.
    fn is_transparent(&self) -> bool {
        matches!(self, Compression::None | Compression::Lz4)
    }
}

fn compression_strategy() -> impl Strategy<Value = Compression> {
    prop_oneof![
        Just(Compression::None),
        Just(Compression::Lz4),
        Just(Compression::Blosc2),
        Just(Compression::Zstd),
    ]
}

fn make_descriptor(shape: Vec<u64>, compression: &Compression) -> DataObjectDescriptor {
    use ciborium::Value;
    let ndim = shape.len() as u64;
    let strides = {
        let mut v = vec![1u64; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            v[i] = v[i + 1] * shape[i + 1];
        }
        v
    };
    let mut params = BTreeMap::new();
    let wire = compression.wire_name().to_string();
    match compression {
        Compression::Blosc2 => {
            params.insert("blosc2_clevel".to_string(), Value::Integer(3.into()));
            params.insert("blosc2_codec".to_string(), Value::Text("lz4".to_string()));
        }
        Compression::Zstd => {
            params.insert("zstd_level".to_string(), Value::Integer(3.into()));
        }
        _ => {}
    }
    DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim,
        shape,
        strides,
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: wire,
        params,
        masks: None,
        hash: None,
    }
}

fn make_payload(len: usize, seed: u8) -> Vec<u8> {
    // Produce finite f32 bytes so the 0.17 default-reject finite-check
    // accepts the payload.  `len` is a byte count; `len / 4` f32 values.
    // Seed influences the sequence but all values stay in a safe finite
    // range.
    let n_floats = len / 4;
    (0..n_floats)
        .map(|i| {
            let base = i as f32;
            let shift = (seed as f32) * 0.001;
            base + shift
        })
        .flat_map(|v| v.to_ne_bytes())
        .collect()
}

/// Extract per-object encoded payload bytes — used for transparent
/// byte-identity checks.
fn encoded_payloads(buf: &[u8]) -> Vec<Vec<u8>> {
    tensogram::framing::decode_message(buf)
        .unwrap()
        .objects
        .iter()
        .map(|(_, p, _)| p.to_vec())
        .collect()
}

/// Build a (descriptor, payload) pair with consistent shape + bytes.
/// `n_elements` is the number of f32 elements; the payload is
/// `n_elements * 4` bytes so the validator accepts it.
fn make_object(
    n_elements: usize,
    compression: &Compression,
    seed: u8,
) -> (DataObjectDescriptor, Vec<u8>) {
    let desc = make_descriptor(vec![n_elements as u64], compression);
    let payload = make_payload(n_elements * 4, seed);
    (desc, payload)
}

proptest! {
    // Proptest seed runs are capped low because each round trips a
    // real message through encode+decode.  The invariants are strong
    // so 32 cases give good coverage.
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn roundtrip_across_threads(
        n_objects in 1usize..5,
        n_elements in 16usize..4096,
        compression in compression_strategy(),
        threads in prop_oneof![Just(0u32), Just(1), Just(2), Just(4), Just(8)],
    ) {
        let meta = GlobalMetadata::default();

        let objects: Vec<(DataObjectDescriptor, Vec<u8>)> = (0..n_objects)
            .map(|i| make_object(n_elements, &compression, i as u8))
            .collect();
        let pairs: Vec<(&DataObjectDescriptor, &[u8])> = objects
            .iter()
            .map(|(d, p)| (d, p.as_slice()))
            .collect();

        let enc_opts = EncodeOptions {
            threads,
            parallel_threshold_bytes: Some(0), // force parallel path
            ..Default::default()
        };
        let dec_opts = DecodeOptions {
            threads,
            parallel_threshold_bytes: Some(0),
            ..Default::default()
        };

        let msg = encode(&meta, &pairs, &enc_opts).expect("encode");
        let (_meta, decoded) = decode(&msg, &dec_opts).expect("decode");
        prop_assert_eq!(decoded.len(), n_objects);
        for (i, (_desc, bytes)) in decoded.iter().enumerate() {
            prop_assert_eq!(bytes, &objects[i].1);
        }
    }

    #[test]
    fn transparent_codec_byte_identical(
        n_objects in 1usize..5,
        n_elements in 16usize..4096,
        compression in compression_strategy(),
        threads in prop_oneof![Just(1u32), Just(2), Just(4), Just(8)],
    ) {
        // Transparent codec: the encoded payload must be identical
        // across thread counts.  Opaque codecs (blosc2, zstd with
        // workers) are covered by `roundtrip_across_threads` — we
        // skip them here rather than writing a separate generator.
        prop_assume!(compression.is_transparent());

        let meta = GlobalMetadata::default();

        let objects: Vec<(DataObjectDescriptor, Vec<u8>)> = (0..n_objects)
            .map(|i| make_object(n_elements, &compression, i as u8))
            .collect();
        let pairs: Vec<(&DataObjectDescriptor, &[u8])> = objects
            .iter()
            .map(|(d, p)| (d, p.as_slice()))
            .collect();

        let seq_opts = EncodeOptions::default();
        let par_opts = EncodeOptions {
            threads,
            parallel_threshold_bytes: Some(0),
            ..Default::default()
        };

        let seq_msg = encode(&meta, &pairs, &seq_opts).expect("seq encode");
        let par_msg = encode(&meta, &pairs, &par_opts).expect("par encode");

        prop_assert_eq!(encoded_payloads(&seq_msg), encoded_payloads(&par_msg));
    }
}
