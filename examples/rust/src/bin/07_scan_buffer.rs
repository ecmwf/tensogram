// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Example 07 — Scanning a multi-message buffer
//!
//! A .tgm file (or any byte buffer) is a flat sequence of independent messages.
//! scan() finds each message's (offset, length) by looking for TENSOGRM markers
//! and cross-checking against the terminator — it skips corrupt regions.
//!
//! This example:
//!   - Appends 5 messages to a buffer with different metadata
//!   - Deliberately injects garbage bytes between messages 2 and 3
//!   - Scans the buffer and shows recovery from the corrupt region
//!   - Decodes each valid message using its scanned offset

use std::collections::BTreeMap;

use ciborium::Value;
use tensogram::{
    ByteOrder, DataObjectDescriptor, Dtype, EncodeOptions, GlobalMetadata, decode_metadata, encode,
    scan,
};

fn make_message(param: &str, step: i64) -> Vec<u8> {
    let mars = Value::Map(vec![
        (Value::Text("param".into()), Value::Text(param.into())),
        (Value::Text("step".into()), Value::Integer(step.into())),
    ]);
    let mut entry = BTreeMap::new();
    entry.insert("mars".to_string(), mars);

    let global_meta = GlobalMetadata {
        version: 3,
        base: vec![entry],
        ..Default::default()
    };

    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![10],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    };

    let data = vec![0u8; 10 * 4];
    encode(&global_meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Build a buffer with 5 messages ─────────────────────────────────────
    let params_steps = [("2t", 0), ("10u", 0), ("2t", 6), ("10u", 6), ("msl", 0)];
    let messages: Vec<Vec<u8>> = params_steps
        .iter()
        .map(|(p, s)| make_message(p, *s))
        .collect();

    let total_clean: usize = messages.iter().map(|m| m.len()).sum();
    println!("5 messages, {} bytes total (clean)", total_clean);

    // ── 2. Inject corruption between messages 1 and 2 ─────────────────────────
    //
    // Garbage bytes in the middle of a file should not prevent reading the
    // messages that come after. scan() recovers by skipping bytes one at a time
    // when the terminator check fails.
    let mut corrupted_buffer: Vec<u8> = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        corrupted_buffer.extend_from_slice(msg);
        if i == 1 {
            // Insert 256 bytes of garbage
            corrupted_buffer.extend_from_slice(&[0xDEu8; 256]);
            println!("  (injected 256 garbage bytes after message 1)");
        }
    }
    println!(
        "Buffer size with corruption: {} bytes\n",
        corrupted_buffer.len()
    );

    // ── 3. Scan ────────────────────────────────────────────────────────────────
    //
    // scan() returns (start_offset, length) for each valid message found.
    // Corrupted regions produce no entry — they are silently skipped.
    let offsets = scan(&corrupted_buffer);

    println!("scan() found {} valid messages:", offsets.len());
    assert_eq!(offsets.len(), 5, "expected all 5 messages to be found");

    // ── 4. Decode each message from its scanned offset ─────────────────────────
    for (i, (start, len)) in offsets.iter().enumerate() {
        let msg = &corrupted_buffer[*start..*start + *len];
        let meta = decode_metadata(msg)?;

        // Pull the per-object namespace map off base[0] and read two keys.
        let ns = meta.base.first().and_then(|e| e.get("mars"));
        let (param, step) = if let Some(Value::Map(entries)) = ns {
            let param = entries
                .iter()
                .find(|(k, _)| matches!(k, Value::Text(s) if s == "param"))
                .and_then(|(_, v)| {
                    if let Value::Text(t) = v {
                        Some(t.as_str())
                    } else {
                        None
                    }
                })
                .unwrap_or("?");
            let step = entries
                .iter()
                .find(|(k, _)| matches!(k, Value::Text(s) if s == "step"))
                .and_then(|(_, v)| {
                    if let Value::Integer(i) = v {
                        let n: i128 = (*i).into();
                        Some(n)
                    } else {
                        None
                    }
                })
                .unwrap_or(-1);
            (param, step)
        } else {
            ("?", -1)
        };

        println!("  [{i}] offset={start:6}  len={len:6}  param={param:5}  step={step}");
    }

    println!("\nAll 5 messages decoded correctly despite 256 bytes of injected garbage.");
    Ok(())
}
