//! Example 10 — Iterator APIs
//!
//! Tensogram provides three iterator types for ergonomic traversal:
//!
//!   messages(&buf)   — zero-copy iterator over messages in a byte buffer
//!   objects(&msg, _) — lazy iterator decoding each object on demand
//!   file.iter()      — lazy seek-based iteration over file messages
//!
//! All three implement Iterator and ExactSizeIterator.

use std::collections::BTreeMap;

use ciborium::Value;
use tensogram_core::{
    decode, messages, objects, objects_metadata, ByteOrder, DataObjectDescriptor, DecodeOptions,
    Dtype, EncodeOptions, GlobalMetadata, TensogramFile,
};

fn make_message(param: &str, fill: u8) -> (GlobalMetadata, DataObjectDescriptor, Vec<u8>) {
    let mars = Value::Map(vec![(
        Value::Text("param".into()),
        Value::Text(param.into()),
    )]);
    let mut entry = BTreeMap::new();
    entry.insert("mars".to_string(), mars);

    let global_meta = GlobalMetadata {
        version: 2,
        base: vec![entry],
        ..Default::default()
    };

    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 2,
        shape: vec![4u64, 4],
        strides: vec![4u64, 1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    };

    let data = vec![fill; 4 * 4 * 4]; // 4×4 float32
    (global_meta, desc, data)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Buffer message iteration ─────────────────────────────────────────
    //
    // Encode 3 messages, concatenate into a single buffer, then iterate.
    let params = ["2t", "10u", "msl"];
    let encoded: Vec<Vec<u8>> = params
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let (meta, desc, data) = make_message(p, i as u8);
            tensogram_core::encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap()
        })
        .collect();

    let mut buf = Vec::new();
    for msg in &encoded {
        buf.extend_from_slice(msg);
    }

    println!("=== Buffer message iterator ===");
    let iter = messages(&buf);
    println!("  {} messages found (ExactSizeIterator)", iter.len());

    for (i, msg_slice) in messages(&buf).enumerate() {
        let (meta, _) = decode(msg_slice, &DecodeOptions::default())?;
        let param = meta
            .base
            .first()
            .and_then(|e| e.get("mars"))
            .map(|m| get_text(m, "param"))
            .unwrap_or("?");
        println!("  [{i}] param={param}  bytes={}", msg_slice.len());
    }

    // ── 2. Object iteration ─────────────────────────────────────────────────
    //
    // Iterate over the decoded objects inside a single message.
    println!("\n=== Object iterator ===");
    let first_msg = messages(&buf).next().unwrap();
    for result in objects(first_msg, DecodeOptions::default())? {
        let (desc, data) = result?;
        println!(
            "  shape={:?}  dtype={}  data={} bytes",
            desc.shape,
            desc.dtype,
            data.len()
        );
    }

    // ── 3. Metadata-only iteration ──────────────────────────────────────────
    //
    // Get descriptors without decoding any payload data.
    println!("\n=== Metadata-only ===");
    for desc in objects_metadata(first_msg)? {
        println!(
            "  ndim={}  shape={:?}  dtype={}",
            desc.ndim, desc.shape, desc.dtype
        );
    }

    // ── 4. File-based iteration ─────────────────────────────────────────────
    //
    // Write messages to a file, then iterate lazily with seek-based I/O.
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("iter_demo.tgm");

    {
        let mut file = TensogramFile::create(&path)?;
        for (i, &p) in params.iter().enumerate() {
            let (meta, desc, data) = make_message(p, i as u8);
            file.append(&meta, &[(&desc, &data)], &EncodeOptions::default())?;
        }
    }

    println!("\n=== File iterator ===");
    let file = TensogramFile::open(&path)?;
    let file_iter = file.iter()?;
    println!("  {} messages in file (ExactSizeIterator)", file_iter.len());

    for (i, raw) in file_iter.enumerate() {
        let raw = raw?;
        let (meta, _) = decode(&raw, &DecodeOptions::default())?;
        let param = meta
            .base
            .first()
            .and_then(|e| e.get("mars"))
            .map(|m| get_text(m, "param"))
            .unwrap_or("?");
        println!("  [{i}] param={param}  bytes={}", raw.len());

        // Nested: iterate objects within this message
        for result in objects(&raw, DecodeOptions::default())? {
            let (desc, data) = result?;
            println!(
                "       shape={:?}  dtype={}  {} bytes",
                desc.shape,
                desc.dtype,
                data.len()
            );
        }
    }

    println!("\nIterator example complete.");
    Ok(())
}

fn get_text<'a>(map: &'a Value, key: &str) -> &'a str {
    if let Value::Map(entries) = map {
        for (k, v) in entries {
            if matches!(k, Value::Text(s) if s == key) {
                if let Value::Text(t) = v {
                    return t;
                }
            }
        }
    }
    "?"
}
