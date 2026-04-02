//! Example 06 — Payload integrity hashing
//!
//! Every object payload can carry a hash of its encoded bytes.
//! This detects corruption introduced during storage or transmission.
//!
//! EncodeOptions::default() uses xxh3 (fast, 64-bit non-cryptographic).
//! sha1 and md5 are available for archival or legacy compatibility.
//!
//! The hash is stored in the CBOR metadata alongside the payload descriptor,
//! so it survives any transport layer that carries the full message bytes.

use std::collections::BTreeMap;

use tensogram_core::{
    decode, encode, ByteOrder, DecodeOptions, Dtype, EncodeOptions, HashAlgorithm, Metadata,
    ObjectDescriptor, PayloadDescriptor, TensogramError,
};

fn make_metadata() -> Metadata {
    Metadata {
        version: 1,
        objects: vec![ObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 1,
            shape: vec![100],
            strides: vec![1],
            dtype: Dtype::Float32,
            extra: BTreeMap::new(),
        }],
        payload: vec![PayloadDescriptor {
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        }],
        extra: BTreeMap::new(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = vec![42u8; 100 * 4];
    let metadata = make_metadata();

    // ── 1. Encode with xxh3 (default) ─────────────────────────────────────────
    {
        let options = EncodeOptions {
            hash_algorithm: Some(HashAlgorithm::Xxh3),
        };
        let message = encode(&metadata, &[&data], &options)?;

        let meta = tensogram_core::decode_metadata(&message)?;
        let hash = meta.payload[0].hash.as_ref().unwrap();
        println!("xxh3 hash: {}:{}", hash.hash_type, hash.value);

        // Verify on decode — this is where the hash is checked
        let verify_opts = DecodeOptions { verify_hash: true };
        decode(&message, &verify_opts)?;
        println!("xxh3 verification: PASS");
    }

    // ── 2. Encode with sha1 ────────────────────────────────────────────────────
    {
        let options = EncodeOptions {
            hash_algorithm: Some(HashAlgorithm::Sha1),
        };
        let message = encode(&metadata, &[&data], &options)?;

        let meta = tensogram_core::decode_metadata(&message)?;
        let hash = meta.payload[0].hash.as_ref().unwrap();
        println!("\nsha1 hash: {}:{}", hash.hash_type, &hash.value[..16]);

        let verify_opts = DecodeOptions { verify_hash: true };
        decode(&message, &verify_opts)?;
        println!("sha1 verification: PASS");
    }

    // ── 3. Encode with md5 ─────────────────────────────────────────────────────
    {
        let options = EncodeOptions {
            hash_algorithm: Some(HashAlgorithm::Md5),
        };
        let message = encode(&metadata, &[&data], &options)?;

        let meta = tensogram_core::decode_metadata(&message)?;
        let hash = meta.payload[0].hash.as_ref().unwrap();
        println!("\nmd5 hash:  {}:{}", hash.hash_type, &hash.value[..16]);

        let verify_opts = DecodeOptions { verify_hash: true };
        decode(&message, &verify_opts)?;
        println!("md5 verification: PASS");
    }

    // ── 4. Encode with no hash ─────────────────────────────────────────────────
    {
        let options = EncodeOptions {
            hash_algorithm: None,
        };
        let message = encode(&metadata, &[&data], &options)?;

        let meta = tensogram_core::decode_metadata(&message)?;
        assert!(meta.payload[0].hash.is_none());

        // verify_hash: true on a message without a hash is silently skipped
        let verify_opts = DecodeOptions { verify_hash: true };
        decode(&message, &verify_opts)?;
        println!("\nNo-hash message decoded with verify_hash=true: silently skipped (OK)");
    }

    // ── 5. Corruption detection ────────────────────────────────────────────────
    {
        let message = encode(&metadata, &[&data], &EncodeOptions::default())?;

        // Corrupt one byte in the payload area (past the CBOR metadata section)
        let mut corrupted = message.clone();
        // Find OBJS marker and flip a byte in the payload
        let objs_pos = corrupted
            .windows(4)
            .position(|w| w == b"OBJS")
            .expect("OBJS marker not found");
        corrupted[objs_pos + 10] ^= 0xFF; // flip a byte

        let verify_opts = DecodeOptions { verify_hash: true };
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
