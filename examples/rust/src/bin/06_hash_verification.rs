// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Example 06 — Payload integrity hashing
//!
//! Every object payload can carry a hash of its encoded bytes.
//! This detects corruption introduced during storage or transmission.
//!
//! EncodeOptions::default() uses xxh3 (fast, 64-bit non-cryptographic).
//!
//! The hash is stored in the CBOR metadata alongside the object descriptor,
//! so it survives any transport layer that carries the full message bytes.

use std::collections::BTreeMap;

use tensogram::{
    ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions, GlobalMetadata,
    HashAlgorithm, TensogramError, decode, encode,
};

fn make_descriptor() -> DataObjectDescriptor {
    DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![100],
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

fn make_global_meta() -> GlobalMetadata {
    GlobalMetadata {
        version: 2,
        extra: BTreeMap::new(),
        ..Default::default()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = vec![42u8; 100 * 4];
    let desc = make_descriptor();
    let global_meta = make_global_meta();

    // ── 1. Encode with xxh3 (default) ─────────────────────────────────────────
    {
        let options = EncodeOptions {
            hash_algorithm: Some(HashAlgorithm::Xxh3),
            ..Default::default()
        };
        let message = encode(&global_meta, &[(&desc, &data)], &options)?;

        let (_, objects) = decode(&message, &DecodeOptions::default())?;
        let hash = objects[0].0.hash.as_ref().unwrap();
        println!("xxh3 hash: {}:{}", hash.hash_type, hash.value);

        // Verify on decode — this is where the hash is checked
        let verify_opts = DecodeOptions {
            verify_hash: true,
            ..Default::default()
        };
        decode(&message, &verify_opts)?;
        println!("xxh3 verification: PASS");
    }

    // ── 2. Encode with no hash ──────────────────────────────────────────────────
    {
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let message = encode(&global_meta, &[(&desc, &data)], &options)?;

        let (_, objects) = decode(&message, &DecodeOptions::default())?;
        assert!(objects[0].0.hash.is_none());

        // verify_hash: true on a message without a hash is silently skipped
        let verify_opts = DecodeOptions {
            verify_hash: true,
            ..Default::default()
        };
        decode(&message, &verify_opts)?;
        println!("\nNo-hash message decoded with verify_hash=true: silently skipped (OK)");
    }

    // ── 3. Corruption detection ─────────────────────────────────────────────────
    {
        let message = encode(&global_meta, &[(&desc, &data)], &EncodeOptions::default())?;

        // Corrupt one byte in the payload area.
        // The message preamble is 24 bytes, followed by frame headers (16 bytes each).
        // Flip a byte well into the message to hit the encoded payload region.
        let mut corrupted = message.clone();
        let mid = message.len() / 2;
        corrupted[mid] ^= 0xFF; // flip a byte in the middle of the message

        let verify_opts = DecodeOptions {
            verify_hash: true,
            ..Default::default()
        };
        let result = decode(&corrupted, &verify_opts);

        match result {
            Err(TensogramError::HashMismatch { expected, actual }) => {
                println!("\nCorruption detected:");
                println!("  expected hash: {}", &expected[..16]);
                println!("  actual hash:   {}", &actual[..16]);
                println!("  HashMismatch error returned correctly.");
            }
            Ok(_) => panic!("expected hash mismatch error"),
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    Ok(())
}
