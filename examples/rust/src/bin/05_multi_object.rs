// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Example 05 — Multiple objects in one message
//!
//! A single Tensogram message can carry several tensors.  Each object has
//! its own shape, dtype, encoding pipeline, and per-object metadata.
//!
//! Real-world examples:
//!   * A wave-spectrum message carrying a 3-tensor spectrum and a 2-tensor
//!     land/sea mask (the pattern demonstrated below).
//!   * A medical-imaging message carrying a 4-D time-series volume, a 3-D
//!     segmentation mask, and a 1-D array of acquisition timestamps.
//!   * An ML-pipeline message carrying a batch of input features, a label
//!     tensor, and a validity bitmask.
//!
//! This example uses weather metadata (MARS) as concrete context, but the
//! same mechanism works with any application vocabulary.

use std::collections::BTreeMap;

use ciborium::Value;
use tensogram::{
    ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions, GlobalMetadata, decode,
    decode_object, encode,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nlat = 30usize;
    let nlon = 60usize;
    let nfreq = 25usize;

    // ── Object 0: wave spectrum (float32, lat × lon × freq) ──────────────────
    let spectrum: Vec<u8> = (0..(nlat * nlon * nfreq))
        .flat_map(|i| (i as f32 * 0.001).to_ne_bytes())
        .collect();

    // ── Per-object metadata → base[0] ───────────────────────────────────────
    // Each base entry holds ALL metadata for one object independently.
    // Shared keys (class, date, step, type) are repeated in each entry.
    let obj0_mars = Value::Map(vec![
        (Value::Text("class".into()), Value::Text("od".into())),
        (Value::Text("date".into()), Value::Text("20260401".into())),
        (Value::Text("step".into()), Value::Integer(6.into())),
        (Value::Text("type".into()), Value::Text("fc".into())),
        (
            Value::Text("param".into()),
            Value::Text("wave_spectra".into()),
        ),
        (Value::Text("levtype".into()), Value::Text("sfc".into())),
    ]);
    let mut obj0_base = BTreeMap::new();
    obj0_base.insert("mars".to_string(), obj0_mars);

    // ── Object 1: land/sea mask (uint8, lat × lon) ────────────────────────────
    let mask: Vec<u8> = (0..(nlat * nlon))
        .map(|i| if i % 3 == 0 { 0u8 } else { 1u8 }) // alternating land/sea
        .collect();

    let obj1_mars = Value::Map(vec![
        (Value::Text("class".into()), Value::Text("od".into())),
        (Value::Text("date".into()), Value::Text("20260401".into())),
        (Value::Text("step".into()), Value::Integer(6.into())),
        (Value::Text("type".into()), Value::Text("fc".into())),
        (Value::Text("param".into()), Value::Text("lsm".into())),
        (Value::Text("levtype".into()), Value::Text("sfc".into())),
    ]);
    let mut obj1_base = BTreeMap::new();
    obj1_base.insert("mars".to_string(), obj1_mars);

    // ── Descriptors: one per object ───────────────────────────────────────────
    //
    // RULE: descriptors.len() == data_slices.len().
    // Each descriptor pairs with its data slice by index.
    let desc0 = DataObjectDescriptor {
        // object 0 — spectrum
        obj_type: "ntensor".to_string(),
        ndim: 3,
        shape: vec![nlat as u64, nlon as u64, nfreq as u64],
        strides: vec![(nlon * nfreq) as u64, nfreq as u64, 1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };

    let desc1 = DataObjectDescriptor {
        // object 1 — mask
        obj_type: "ntensor".to_string(),
        ndim: 2,
        shape: vec![nlat as u64, nlon as u64],
        strides: vec![nlon as u64, 1],
        dtype: Dtype::Uint8,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };

    let global_meta = GlobalMetadata {
        version: 3,
        base: vec![obj0_base, obj1_base],
        ..Default::default()
    };

    // ── Encode both objects in one message ────────────────────────────────────
    let message = encode(
        &global_meta,
        &[(&desc0, &spectrum), (&desc1, &mask)],
        &EncodeOptions::default(),
    )?;

    println!(
        "Encoded: {} bytes  (spectrum={} bytes  mask={} bytes)",
        message.len(),
        spectrum.len(),
        mask.len()
    );

    // ── Decode all objects ────────────────────────────────────────────────────
    let (_meta, objects) = decode(&message, &DecodeOptions::default())?;

    println!("\ndecode() — all objects:");
    for (i, (obj_desc, obj_data)) in objects.iter().enumerate() {
        println!(
            "  [{i}] dtype={}, shape={:?}, {} bytes decoded",
            obj_desc.dtype,
            obj_desc.shape,
            obj_data.len(),
        );
    }
    assert_eq!(objects[0].1, spectrum);
    assert_eq!(objects[1].1, mask);

    // ── Decode a single object by index (O(1) via binary header) ──────────────
    //
    // The binary header stores each object's byte offset, so the decoder
    // seeks directly to object 1 without reading object 0.
    let (_meta2, _desc1_decoded, mask_decoded) =
        decode_object(&message, 1, &DecodeOptions::default())?;

    println!("\ndecode_object(index=1):");
    println!(
        "  {} bytes, matches original: {}",
        mask_decoded.len(),
        mask_decoded == mask
    );

    Ok(())
}
