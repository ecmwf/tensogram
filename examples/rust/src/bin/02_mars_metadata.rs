//! Example 02 — MARS-namespaced metadata
//!
//! Shows how to attach ECMWF MARS vocabulary keys to a message and to
//! individual objects, then read them back after decoding.
//!
//! The library is vocabulary-agnostic: it stores and returns whatever keys
//! you put in. Meaning is assigned by the application layer.

use std::collections::BTreeMap;

use ciborium::Value;
use tensogram_core::{
    decode, decode_metadata, encode,
    ByteOrder, DecodeOptions, Dtype, EncodeOptions,
    Metadata, ObjectDescriptor, PayloadDescriptor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Message-level MARS keys ───────────────────────────────────────────────
    //
    // Convention: group keys under a namespace map, e.g. "mars".
    // The library sorts map keys canonically (RFC 8949 §4.2) on encode,
    // so insertion order does not matter.
    let mars_msg = Value::Map(vec![
        (Value::Text("class".into()), Value::Text("od".into())),
        (Value::Text("date".into()),  Value::Text("20260401".into())),
        (Value::Text("step".into()),  Value::Integer(6.into())),
        (Value::Text("time".into()),  Value::Text("0000".into())),
        (Value::Text("type".into()),  Value::Text("fc".into())),
    ]);

    let mut msg_extra = BTreeMap::new();
    msg_extra.insert("mars".to_string(), mars_msg);

    // ── Per-object MARS keys ──────────────────────────────────────────────────
    //
    // The parameter name lives on the object because different objects in the
    // same message can have different parameters.
    let mars_obj = Value::Map(vec![
        (Value::Text("levtype".into()), Value::Text("sfc".into())),
        (Value::Text("param".into()),   Value::Text("2t".into())),
    ]);

    let mut obj_extra = BTreeMap::new();
    obj_extra.insert("mars".to_string(), mars_obj);

    // ── Build metadata ────────────────────────────────────────────────────────
    let metadata = Metadata {
        version: 1,
        objects: vec![ObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 2,
            shape: vec![721, 1440],    // 0.25-degree global grid
            strides: vec![1440, 1],
            dtype: Dtype::Float32,
            extra: obj_extra,           // parameter goes here
        }],
        payload: vec![PayloadDescriptor {
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        }],
        extra: msg_extra,               // forecast context goes here
    };

    let data = vec![0u8; 721 * 1440 * 4]; // zeros stand in for real values
    let message = encode(&metadata, &[&data], &EncodeOptions::default())?;

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
                    if let Value::Text(t) = v { return Some(t); }
                }
            }
        }
        None
    }

    let mars = &meta.extra["mars"];
    println!("Message-level:");
    println!("  class = {}", get_mars_key(mars, "class").unwrap_or("?"));
    println!("  date  = {}", get_mars_key(mars, "date").unwrap_or("?"));
    println!("  type  = {}", get_mars_key(mars, "type").unwrap_or("?"));
    println!("  step  = {:?}", {
        if let Value::Map(e) = mars {
            e.iter().find(|(k,_)| matches!(k, Value::Text(s) if s == "step"))
             .map(|(_,v)| v)
        } else { None }
    });

    let obj_mars = &meta.objects[0].extra["mars"];
    println!("Object 0:");
    println!("  param   = {}", get_mars_key(obj_mars, "param").unwrap_or("?"));
    println!("  levtype = {}", get_mars_key(obj_mars, "levtype").unwrap_or("?"));
    println!("  shape   = {:?}", meta.objects[0].shape);

    // ── Full decode ───────────────────────────────────────────────────────────
    let (_meta2, objects) = decode(&message, &DecodeOptions::default())?;
    assert_eq!(objects[0], data);
    println!("\nFull decode OK.");

    Ok(())
}
