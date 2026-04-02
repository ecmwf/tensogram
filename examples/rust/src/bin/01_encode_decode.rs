//! Example 01 — Basic encode / decode round-trip
//!
//! Shows the simplest possible usage: describe a tensor, encode it to a
//! self-contained message buffer, decode it back.
//!
//! Run:
//!   cargo run --example 01_encode_decode   (from workspace root)
//!   cargo run --bin 01_encode_decode       (from examples/rust/)

use std::collections::BTreeMap;

use tensogram_core::{
    decode, encode,
    ByteOrder, DecodeOptions, Dtype, EncodeOptions,
    Metadata, ObjectDescriptor, PayloadDescriptor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Describe the tensor ────────────────────────────────────────────────
    //
    // A 100×200 grid of float32 values (e.g. a temperature field on a 0.1-degree
    // lat/lon grid clipped to a small region).
    let shape: Vec<u64> = vec![100, 200];

    // C-contiguous (row-major) strides: advancing along axis 0 skips 200 elements,
    // advancing along axis 1 skips 1 element.
    let strides: Vec<u64> = vec![200, 1];

    let metadata = Metadata {
        version: 1,
        objects: vec![ObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 2,
            shape: shape.clone(),
            strides,
            dtype: Dtype::Float32,
            extra: BTreeMap::new(), // no per-object extra keys in this example
        }],
        payload: vec![PayloadDescriptor {
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),     // exact copy, no quantization
            filter: "none".to_string(),        // no byte rearrangement
            compression: "none".to_string(),   // no compression
            params: BTreeMap::new(),
            hash: None, // hash is filled in by encode() when EncodeOptions::default()
        }],
        extra: BTreeMap::new(),
    };

    // ── 2. Prepare raw data bytes ─────────────────────────────────────────────
    //
    // In production this would come from a model output buffer.
    // Here we generate 100×200 = 20,000 float32 values and serialize them
    // as big-endian bytes to match the payload descriptor.
    let raw_bytes: Vec<u8> = (0u32..20_000)
        .flat_map(|i| {
            let value = 273.15f32 + (i as f32) * 0.001;
            value.to_be_bytes() // big-endian, matching byte_order above
        })
        .collect();

    println!("Input:  {} bytes ({} float32 elements)", raw_bytes.len(), 20_000);

    // ── 3. Encode ─────────────────────────────────────────────────────────────
    //
    // encode() validates lengths, runs the pipeline (encoding → filter → compression),
    // computes an xxh3 hash for each payload, serialises metadata to canonical CBOR,
    // and assembles the wire-format frame.
    let message = encode(&metadata, &[&raw_bytes], &EncodeOptions::default())?;

    println!(
        "Message: {} bytes  (magic={:?}  terminator={:?})",
        message.len(),
        std::str::from_utf8(&message[0..8]).unwrap(),
        std::str::from_utf8(&message[message.len() - 8..]).unwrap(),
    );

    // ── 4. Decode ─────────────────────────────────────────────────────────────
    //
    // decode() parses the binary header, reads CBOR metadata, and for each
    // object: verifies the hash (if verify_hash is set), runs the inverse
    // pipeline, and returns raw bytes in the logical dtype.
    let (decoded_meta, decoded_objects) = decode(&message, &DecodeOptions::default())?;

    println!(
        "Decoded: version={}, {} object(s)",
        decoded_meta.version,
        decoded_objects.len()
    );
    println!(
        "  Object 0: dtype={}, shape={:?}",
        decoded_meta.objects[0].dtype,
        decoded_meta.objects[0].shape,
    );

    assert_eq!(decoded_objects[0], raw_bytes, "round-trip mismatch");
    println!("Round-trip OK: decoded bytes match original.");

    Ok(())
}
