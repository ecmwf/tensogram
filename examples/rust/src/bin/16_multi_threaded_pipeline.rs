// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Example 16 — Multi-threaded coding pipeline (v0.13.0)
//!
//! Demonstrates the caller-controlled `threads` budget added in v0.13.0.
//!
//! Key invariants shown:
//!
//! 1. `threads=0` (default) matches the sequential path byte-identically.
//! 2. Transparent codecs (simple_packing, szip, ...) produce
//!    byte-identical encoded payloads across any `threads` value.
//! 3. Opaque codecs (blosc2, zstd with workers) round-trip losslessly
//!    regardless of thread count.
//!
//! Run with:
//!   cargo run --release --bin 16_multi_threaded_pipeline

use std::collections::BTreeMap;
use std::time::Instant;

use tensogram::{
    ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions, GlobalMetadata, decode,
    encode, framing,
};

fn encoded_payloads(msg: &[u8]) -> Vec<Vec<u8>> {
    let decoded = framing::decode_message(msg).expect("decode_message");
    decoded
        .objects
        .iter()
        .map(|(_, payload, _)| payload.to_vec())
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Build a large single-object message ────────────────────────────
    //
    // 16 million f64 values (≈ 122 MiB) — representative of an ML output
    // or an atmospheric field at operational resolution.

    let n = 16_000_000;
    let values: Vec<f64> = (0..n).map(|i| 250.0 + (i as f64).sin() * 30.0).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let meta = GlobalMetadata::default();
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![n as u64],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    };

    // ── 2. Sequential (threads=0) vs parallel (threads=8) ───────────────────
    //
    // For a transparent pipeline the encoded payload must be
    // byte-identical across thread counts.

    let opts_seq = EncodeOptions::default();
    let opts_par = EncodeOptions {
        threads: 8,
        ..Default::default()
    };

    let t0 = Instant::now();
    let msg_seq = encode(&meta, &[(&desc, &data)], &opts_seq)?;
    let dur_seq = t0.elapsed();

    let t0 = Instant::now();
    let msg_par = encode(&meta, &[(&desc, &data)], &opts_par)?;
    let dur_par = t0.elapsed();

    println!(
        "Encode {} f64 values (= {:.1} MiB):",
        n,
        data.len() as f64 / (1024.0 * 1024.0)
    );
    println!("  threads = 0: {:>7.1} ms", dur_seq.as_secs_f64() * 1000.0);
    println!(
        "  threads = 8: {:>7.1} ms   (x{:.2} speedup)",
        dur_par.as_secs_f64() * 1000.0,
        dur_seq.as_secs_f64() / dur_par.as_secs_f64().max(1e-9),
    );

    // Transparent pipeline invariant: encoded payloads identical.
    let seq_payloads = encoded_payloads(&msg_seq);
    let par_payloads = encoded_payloads(&msg_par);
    assert_eq!(
        seq_payloads, par_payloads,
        "transparent pipeline must be byte-identical across thread counts"
    );
    println!("  ✓ encoded payloads are byte-identical across threads");

    // ── 3. Decode with a thread budget — same output, different path ────
    let opts_dec_seq = DecodeOptions::default();
    let opts_dec_par = DecodeOptions {
        threads: 8,
        ..Default::default()
    };

    let t0 = Instant::now();
    let (_, dec_seq) = decode(&msg_seq, &opts_dec_seq)?;
    let dec_seq_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = Instant::now();
    let (_, dec_par) = decode(&msg_seq, &opts_dec_par)?;
    let dec_par_ms = t0.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(dec_seq[0].1, dec_par[0].1, "decoded bytes must match");
    println!();
    println!("Decode (threads=0): {:>7.1} ms", dec_seq_ms);
    println!(
        "Decode (threads=8): {:>7.1} ms   (x{:.2} speedup)",
        dec_par_ms,
        dec_seq_ms / dec_par_ms.max(1e-9),
    );

    // ── 4. Opaque codec — round-trip losslessly, may differ byte-wise ──
    //
    // blosc2 with nthreads > 0 produces a different compressed byte
    // stream (blocks land in worker completion order) but always
    // round-trips to the same data.  blosc2 is part of the default
    // tensogram feature set, so no cfg gate is needed here.

    use ciborium::Value;
    let mut blosc2_desc = desc.clone();
    blosc2_desc.compression = "blosc2".to_string();
    blosc2_desc
        .params
        .insert("blosc2_clevel".to_string(), Value::Integer(5.into()));
    blosc2_desc
        .params
        .insert("blosc2_codec".to_string(), Value::Text("lz4".to_string()));

    let msg_b0 = encode(&meta, &[(&blosc2_desc, &data)], &opts_seq)?;
    let msg_b8 = encode(&meta, &[(&blosc2_desc, &data)], &opts_par)?;

    let (_, dec_b0) = decode(&msg_b0, &opts_dec_seq)?;
    let (_, dec_b8) = decode(&msg_b8, &opts_dec_seq)?;

    println!();
    println!("blosc2 opaque pipeline:");
    println!("  encode bytes differ across threads: {}", msg_b0 != msg_b8);
    println!("  decoded data matches: {}", dec_b0[0].1 == dec_b8[0].1);
    assert_eq!(
        dec_b0[0].1, dec_b8[0].1,
        "blosc2 must round-trip losslessly regardless of threads"
    );

    Ok(())
}
