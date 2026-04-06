//! Example 02 — MARS-namespaced metadata
//!
//! Shows how to attach ECMWF MARS vocabulary keys to a message and to
//! individual objects, then read them back after decoding.
//!
//! - Per-object MARS keys live in `base[i]["mars"]`.
//! - Each base entry holds ALL metadata for that object independently.
//!
//! The library is vocabulary-agnostic: it stores and returns whatever keys
//! you put in. Meaning is assigned by the application layer.

use std::collections::BTreeMap;

use ciborium::Value;
use tensogram_core::{
    decode, decode_metadata, encode, ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype,
    EncodeOptions, GlobalMetadata,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Per-object MARS keys → base[0]["mars"] ────────────────────────────────
    //
    // Each base entry holds ALL metadata for one object.  Keys shared across
    // objects are simply repeated in each entry — the library does not track
    // what is common vs varying.
    let mars_obj0 = Value::Map(vec![
        (Value::Text("class".into()), Value::Text("od".into())),
        (Value::Text("date".into()), Value::Text("20260401".into())),
        (Value::Text("step".into()), Value::Integer(6.into())),
        (Value::Text("time".into()), Value::Text("0000".into())),
        (Value::Text("type".into()), Value::Text("fc".into())),
        (Value::Text("levtype".into()), Value::Text("sfc".into())),
        (Value::Text("param".into()), Value::Text("2t".into())),
    ]);

    let mut obj0_base = BTreeMap::new();
    obj0_base.insert("mars".to_string(), mars_obj0);

    // ── Build descriptor and global metadata ──────────────────────────────────
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 2,
        shape: vec![721, 1440], // 0.25-degree global grid
        strides: vec![1440, 1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(), // encoding params only — no MARS keys here
        hash: None,
    };

    let global_meta = GlobalMetadata {
        version: 2,
        base: vec![obj0_base],
        ..Default::default()
    };

    let data = vec![0u8; 721 * 1440 * 4]; // zeros stand in for real values
    let message = encode(&global_meta, &[(&desc, &data)], &EncodeOptions::default())?;

    // ── Read back metadata only (no payload decode) ───────────────────────────
    //
    // decode_metadata() parses the binary header and CBOR section, then stops.
    // No object bytes are read or allocated. Useful for filtering/listing.
    let meta = decode_metadata(&message)?;

    // ── Navigate the MARS namespace ───────────────────────────────────────────
    fn get_mars_key<'a>(map: &'a Value, key: &str) -> Option<&'a str> {
        if let Value::Map(entries) = map {
            for (k, v) in entries {
                if matches!(k, Value::Text(s) if s == key) {
                    if let Value::Text(t) = v {
                        return Some(t);
                    }
                }
            }
        }
        None
    }

    let mars = &meta.base[0]["mars"];
    println!("Object 0 (base mars):");
    println!("  class   = {}", get_mars_key(mars, "class").unwrap_or("?"));
    println!("  date    = {}", get_mars_key(mars, "date").unwrap_or("?"));
    println!("  type    = {}", get_mars_key(mars, "type").unwrap_or("?"));
    println!("  step    = {:?}", {
        if let Value::Map(e) = mars {
            e.iter()
                .find(|(k, _)| matches!(k, Value::Text(s) if s == "step"))
                .map(|(_, v)| v)
        } else {
            None
        }
    });

    // ── Full decode ───────────────────────────────────────────────────────────
    let (meta2, objects) = decode(&message, &DecodeOptions::default())?;

    // Per-object MARS keys are in base[i]["mars"]
    let obj_mars = &meta2.base[0]["mars"];
    println!("Object 0 (decoded mars):");
    println!(
        "  param   = {}",
        get_mars_key(obj_mars, "param").unwrap_or("?")
    );
    println!(
        "  levtype = {}",
        get_mars_key(obj_mars, "levtype").unwrap_or("?")
    );
    println!("  shape   = {:?}", objects[0].0.shape);

    assert_eq!(objects[0].1, data);
    println!("\nFull decode OK.");

    Ok(())
}
