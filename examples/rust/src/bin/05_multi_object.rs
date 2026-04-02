//! Example 05 — Multiple objects in one message
//!
//! A single Tensogram message can carry several tensors.  Each object has
//! its own shape, dtype, encoding pipeline, and per-object metadata.
//!
//! Real-world use case: a sea wave spectrum message carries
//!   Object 0 — the wave spectrum proper (float32, 3-tensor: lat × lon × freq)
//!   Object 1 — land/sea mask (uint8, 2-tensor: lat × lon)
//!
//! Both share the same forecast context metadata (date, step, domain).

use std::collections::BTreeMap;

use ciborium::Value;
use tensogram_core::{
    decode, decode_object, encode,
    ByteOrder, DecodeOptions, Dtype, EncodeOptions,
    Metadata, ObjectDescriptor, PayloadDescriptor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nlat = 30usize;
    let nlon = 60usize;
    let nfreq = 25usize;

    // ── Object 0: wave spectrum (float32, lat × lon × freq) ──────────────────
    let spectrum: Vec<u8> = (0..(nlat * nlon * nfreq))
        .flat_map(|i| (i as f32 * 0.001).to_be_bytes())
        .collect();

    let obj0_mars = Value::Map(vec![
        (Value::Text("param".into()), Value::Text("wave_spectra".into())),
        (Value::Text("levtype".into()), Value::Text("sfc".into())),
    ]);
    let mut obj0_extra = BTreeMap::new();
    obj0_extra.insert("mars".to_string(), obj0_mars);

    // ── Object 1: land/sea mask (uint8, lat × lon) ────────────────────────────
    let mask: Vec<u8> = (0..(nlat * nlon))
        .map(|i| if i % 3 == 0 { 0u8 } else { 1u8 }) // alternating land/sea
        .collect();

    let obj1_mars = Value::Map(vec![
        (Value::Text("param".into()), Value::Text("lsm".into())),
        (Value::Text("levtype".into()), Value::Text("sfc".into())),
    ]);
    let mut obj1_extra = BTreeMap::new();
    obj1_extra.insert("mars".to_string(), obj1_mars);

    // ── Shared forecast context ───────────────────────────────────────────────
    let mars_msg = Value::Map(vec![
        (Value::Text("class".into()), Value::Text("od".into())),
        (Value::Text("date".into()),  Value::Text("20260401".into())),
        (Value::Text("step".into()),  Value::Integer(6.into())),
        (Value::Text("type".into()),  Value::Text("fc".into())),
    ]);
    let mut msg_extra = BTreeMap::new();
    msg_extra.insert("mars".to_string(), mars_msg);

    // ── Metadata: two objects, two payload descriptors ────────────────────────
    //
    // RULE: objects.len() == payload.len() == data_slices.len().
    // They correspond by index.
    let metadata = Metadata {
        version: 1,
        objects: vec![
            ObjectDescriptor {               // object 0 — spectrum
                obj_type: "ntensor".to_string(),
                ndim: 3,
                shape: vec![nlat as u64, nlon as u64, nfreq as u64],
                strides: vec![(nlon * nfreq) as u64, nfreq as u64, 1],
                dtype: Dtype::Float32,
                extra: obj0_extra,
            },
            ObjectDescriptor {               // object 1 — mask
                obj_type: "ntensor".to_string(),
                ndim: 2,
                shape: vec![nlat as u64, nlon as u64],
                strides: vec![nlon as u64, 1],
                dtype: Dtype::Uint8,
                extra: obj1_extra,
            },
        ],
        payload: vec![
            PayloadDescriptor {              // payload 0 — float32 big-endian, no encoding
                byte_order: ByteOrder::Big,
                encoding: "none".to_string(),
                filter: "none".to_string(),
                compression: "none".to_string(),
                params: BTreeMap::new(),
                hash: None,
            },
            PayloadDescriptor {              // payload 1 — uint8, no encoding needed
                byte_order: ByteOrder::Big,
                encoding: "none".to_string(),
                filter: "none".to_string(),
                compression: "none".to_string(),
                params: BTreeMap::new(),
                hash: None,
            },
        ],
        extra: msg_extra,
    };

    // ── Encode both objects in one message ────────────────────────────────────
    let message = encode(&metadata, &[&spectrum, &mask], &EncodeOptions::default())?;

    println!(
        "Encoded: {} bytes  (spectrum={} bytes  mask={} bytes)",
        message.len(), spectrum.len(), mask.len()
    );

    // ── Decode all objects ────────────────────────────────────────────────────
    let (meta, objects) = decode(&message, &DecodeOptions::default())?;

    println!("\ndecode() — all objects:");
    for (i, obj) in meta.objects.iter().enumerate() {
        println!(
            "  [{i}] dtype={}, shape={:?}, {} bytes decoded",
            obj.dtype,
            obj.shape,
            objects[i].len(),
        );
    }
    assert_eq!(objects[0], spectrum);
    assert_eq!(objects[1], mask);

    // ── Decode a single object by index (O(1) via binary header) ──────────────
    //
    // The binary header stores each object's byte offset, so the decoder
    // seeks directly to object 1 without reading object 0.
    let (_meta2, mask_decoded) = decode_object(&message, 1, &DecodeOptions::default())?;

    println!("\ndecode_object(index=1):");
    println!("  {} bytes, matches original: {}", mask_decoded.len(), mask_decoded == mask);

    Ok(())
}
