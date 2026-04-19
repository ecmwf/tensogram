// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Example 02b — Per-object metadata with a generic application namespace
//!
//! Shows that the metadata mechanism in example 02 is not specific to the
//! MARS vocabulary — any application namespace works the same way. Here we
//! use a made-up `"product"` namespace to tag a 2-D field with semantic
//! context (name, units, acquisition device, etc.).
//!
//! The same pattern applies to any domain vocabulary:
//!   - CF conventions (`"cf"`) for climate/atmospheric data
//!   - BIDS (`"bids"`) for neuroimaging datasets
//!   - DICOM (`"dicom"`) for medical imaging
//!   - Custom (`"experiment"`, `"instrument"`, `"run"`, ...)
//!
//! The library never interprets any of these — it simply stores and returns
//! the keys you supply. Meaning is assigned by the application layer.

use std::collections::BTreeMap;

use ciborium::Value;
use tensogram::{
    ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions, GlobalMetadata, decode,
    decode_metadata, encode,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Build a "product" namespace for the object ───────────────────────────
    //
    // This is a made-up vocabulary to illustrate that the library stores any
    // keys you put in.  The namespace wraps all product-level metadata so
    // that multiple vocabularies can coexist in the same message without
    // key collisions.
    let product = Value::Map(vec![
        (Value::Text("name".into()), Value::Text("intensity".into())),
        (Value::Text("units".into()), Value::Text("counts".into())),
        (
            Value::Text("device".into()),
            Value::Text("detector_A".into()),
        ),
        (Value::Text("run_id".into()), Value::Integer(42.into())),
        (
            Value::Text("acquired_at".into()),
            Value::Text("2026-04-18T10:30:00Z".into()),
        ),
    ]);

    // ── Optional: a second parallel namespace for instrument-level metadata ──
    //
    // You can freely add additional namespaces at the same level, and the
    // library stores them all.
    let instrument = Value::Map(vec![
        (Value::Text("serial".into()), Value::Text("XYZ-001".into())),
        (Value::Text("firmware".into()), Value::Text("v3.1.2".into())),
    ]);

    let mut obj0_base = BTreeMap::new();
    obj0_base.insert("product".to_string(), product);
    obj0_base.insert("instrument".to_string(), instrument);

    // ── Build descriptor and global metadata ─────────────────────────────────
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 2,
        shape: vec![512, 512],
        strides: vec![512, 1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
        hash: None,
    };

    let global_meta = GlobalMetadata {
        version: 2,
        base: vec![obj0_base],
        ..Default::default()
    };

    let data = vec![0u8; 512 * 512 * 4]; // zeros stand in for real values
    let message = encode(&global_meta, &[(&desc, &data)], &EncodeOptions::default())?;

    // ── Read back metadata only (no payload decode) ──────────────────────────
    let meta = decode_metadata(&message)?;

    // Helper to pull a text value from a CBOR map by key.
    fn get_text<'a>(map: &'a Value, key: &str) -> Option<&'a str> {
        if let Value::Map(entries) = map {
            for (k, v) in entries {
                if matches!(k, Value::Text(s) if s == key)
                    && let Value::Text(t) = v
                {
                    return Some(t);
                }
            }
        }
        None
    }

    let product_meta = &meta.base[0]["product"];
    println!("Object 0 (product namespace):");
    println!(
        "  name   = {}",
        get_text(product_meta, "name").unwrap_or("?")
    );
    println!(
        "  units  = {}",
        get_text(product_meta, "units").unwrap_or("?")
    );
    println!(
        "  device = {}",
        get_text(product_meta, "device").unwrap_or("?")
    );

    let instrument_meta = &meta.base[0]["instrument"];
    println!("Object 0 (instrument namespace):");
    println!(
        "  serial   = {}",
        get_text(instrument_meta, "serial").unwrap_or("?")
    );
    println!(
        "  firmware = {}",
        get_text(instrument_meta, "firmware").unwrap_or("?")
    );

    // ── Full decode ──────────────────────────────────────────────────────────
    let (_meta2, objects) = decode(&message, &DecodeOptions::default())?;
    println!("\nDecoded shape: {:?}", objects[0].0.shape);
    assert_eq!(objects[0].1, data);

    println!("\nGeneric-namespace round-trip OK.");

    Ok(())
}
