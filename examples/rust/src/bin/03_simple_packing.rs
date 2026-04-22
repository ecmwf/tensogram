// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Example 03 — Simple packing (lossy compression)
//!
//! simple_packing quantises float64 values into N-bit integers (GRIB-style).
//! At 16 bits per value the payload is 4x smaller than float32 and 8x smaller
//! than float64, with precision loss typically below instrument noise for most
//! bounded-range scientific measurements (temperature, pressure, voltage, intensity).
//!
//! Steps:
//!   1. Compute packing parameters from your data.
//!   2. Put the parameters in the DataObjectDescriptor.
//!   3. Pass the raw f64 bytes to encode() — the pipeline does the rest.
//!   4. decode() returns f64 bytes regardless of the original dtype.

use std::collections::BTreeMap;

use ciborium::Value;
use tensogram::{
    ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions, GlobalMetadata, decode,
    encode,
};
use tensogram_encodings::simple_packing;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Source data ────────────────────────────────────────────────────────
    //
    // Simulate a 1D temperature array: 1000 values from ~249 K to ~349 K.
    let n = 1000usize;
    let values: Vec<f64> = (0..n).map(|i| 249.15 + (i as f64) * 0.1).collect();

    // encode() expects raw bytes. simple_packing always operates on f64.
    // Use native-endian bytes to match byte_order: ByteOrder::native() below.
    let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    println!("Source: {} f64 values  raw={} bytes", n, raw_bytes.len());
    println!("  range: [{:.2}, {:.2}]", values[0], values[n - 1]);

    // ── 2. Compute packing parameters ─────────────────────────────────────────
    //
    // compute_params() finds the best binary_scale_factor for the requested
    // bits_per_value. Returns an error if any value is NaN.
    let bits_per_value: u32 = 16;
    let decimal_scale_factor: i32 = 0; // no decimal scaling in this example
    let params = simple_packing::compute_params(&values, bits_per_value, decimal_scale_factor)?;

    println!("\nPacking params ({}bpv):", bits_per_value);
    println!("  reference_value      = {:.6}", params.reference_value);
    println!("  binary_scale_factor  = {}", params.binary_scale_factor);
    println!("  decimal_scale_factor = {}", params.decimal_scale_factor);
    println!("  bits_per_value       = {}", params.bits_per_value);

    // ── 3. Build descriptor with packing parameters ───────────────────────────
    //
    // The parameters must be stored in the params BTreeMap of the
    // DataObjectDescriptor so decode() can reconstruct them.
    let mut packing_params: BTreeMap<String, Value> = BTreeMap::new();
    packing_params.insert(
        "reference_value".into(),
        Value::Float(params.reference_value),
    );
    packing_params.insert(
        "binary_scale_factor".into(),
        Value::Integer((params.binary_scale_factor as i64).into()),
    );
    packing_params.insert(
        "decimal_scale_factor".into(),
        Value::Integer((params.decimal_scale_factor as i64).into()),
    );
    packing_params.insert(
        "bits_per_value".into(),
        Value::Integer((params.bits_per_value as i64).into()),
    );

    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![n as u64],
        strides: vec![1],
        dtype: Dtype::Float64, // source dtype
        byte_order: ByteOrder::native(),
        encoding: "simple_packing".to_string(), // ← key change
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: packing_params,
        masks: None,
    };

    let global_meta = GlobalMetadata {
        version: 3,
        extra: BTreeMap::new(),
        ..Default::default()
    };

    // ── 4. Encode ─────────────────────────────────────────────────────────────
    //
    // encode() calls simple_packing::encode() internally. The stored payload
    // is the packed bits, not the original f64 bytes.
    let message = encode(
        &global_meta,
        &[(&desc, &raw_bytes)],
        &EncodeOptions::default(),
    )?;

    // Approximate payload size: (n * bits_per_value + 7) / 8 bytes
    let expected_payload = (n * bits_per_value as usize).div_ceil(8);
    println!(
        "\nEncoded: {} bytes total  (raw={} → packed≈{} bytes, {:.1}x smaller)",
        message.len(),
        raw_bytes.len(),
        expected_payload,
        raw_bytes.len() as f64 / expected_payload as f64,
    );

    // ── 5. Decode ─────────────────────────────────────────────────────────────
    //
    // decode() returns f64 bytes — simple_packing always decodes to f64
    // regardless of the dtype stored in the object descriptor.
    let (_meta, objects) = decode(&message, &DecodeOptions::default())?;
    let decoded: Vec<f64> = objects[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();

    // ── 6. Measure precision loss ──────────────────────────────────────────────
    let max_err = values
        .iter()
        .zip(decoded.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    let expected_step =
        (values.last().unwrap() - values.first().unwrap()) / ((1u64 << bits_per_value) - 1) as f64;

    println!(
        "\nPrecision: max_error={:.6} K  (quantisation step≈{:.6} K)",
        max_err, expected_step
    );
    assert!(max_err < 0.01, "precision loss too large: {max_err}");
    println!("OK: all values within 0.01 K of original.");

    Ok(())
}
