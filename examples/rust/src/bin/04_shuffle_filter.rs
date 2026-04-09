//! Example 04 — Byte shuffle filter
//!
//! The shuffle filter rearranges multi-byte elements so that all byte-N across
//! all elements are contiguous. This makes float arrays much more compressible
//! because the predictable high bytes are grouped together.
//!
//!   Before: [B0 B1 B2 B3][B0 B1 B2 B3][B0 B1 B2 B3]...
//!   After:  [B0 B0 B0...][B1 B1 B1...][B2 B2 B2...][B3 B3 B3...]
//!
//! The shuffle filter alone does not reduce byte count — it must be paired
//! with a compressor (szip, deflate) to deliver savings. In this example we
//! demonstrate the round-trip correctness; the compression stage is "none"
//! because szip bindings are not yet implemented.

use std::collections::BTreeMap;

use ciborium::Value;
use tensogram_core::{
    decode, encode, ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions,
    GlobalMetadata,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Source data: 256 float32 values ───────────────────────────────────
    let n = 256usize;
    let values: Vec<f32> = (0..n).map(|i| 1000.0f32 + i as f32 * 0.5).collect();
    let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    println!("Source: {} float32 values = {} bytes", n, raw_bytes.len());

    // ── 2. Build descriptor with shuffle filter ───────────────────────────────
    //
    // shuffle_element_size must equal the byte width of the dtype (4 for float32).
    let mut filter_params: BTreeMap<String, Value> = BTreeMap::new();
    filter_params.insert(
        "shuffle_element_size".to_string(),
        Value::Integer(4.into()), // 4 bytes per float32
    );

    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![n as u64],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "shuffle".to_string(), // ← shuffle is the filter stage
        compression: "none".to_string(),
        params: filter_params,
        hash: None,
    };

    let global_meta = GlobalMetadata {
        version: 2,
        extra: BTreeMap::new(),
        ..Default::default()
    };

    // ── 3. Encode (shuffle applied internally) ────────────────────────────────
    let message = encode(
        &global_meta,
        &[(&desc, &raw_bytes)],
        &EncodeOptions::default(),
    )?;
    println!("Encoded: {} bytes", message.len());

    // ── 4. Decode (unshuffle applied internally) ──────────────────────────────
    //
    // decode() applies the inverse pipeline: decompress → unshuffle → decode.
    // You get back the original bytes unchanged.
    let (_meta, objects) = decode(&message, &DecodeOptions::default())?;

    assert_eq!(objects[0].1, raw_bytes, "shuffle round-trip mismatch");
    println!("Round-trip OK: shuffle + unshuffle produced identical bytes.");

    // ── 5. Direct shuffle API ─────────────────────────────────────────────────
    //
    // You can also call the shuffle functions directly for debugging or
    // integrating with other pipelines.
    use tensogram_encodings::shuffle::{shuffle, unshuffle};

    let shuffled = shuffle(&raw_bytes, 4)?;

    for i in 0..n {
        assert_eq!(
            shuffled[i],
            raw_bytes[i * 4],
            "byte-0 mismatch at element {i}"
        );
    }
    println!("Byte-0 contiguity verified after shuffle.");

    let unshuffled = unshuffle(&shuffled, 4)?;
    assert_eq!(unshuffled, raw_bytes);
    println!("Direct unshuffle OK.");

    Ok(())
}
