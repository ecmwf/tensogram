// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Example 09 — File API (TensogramFile)
//!
//! TensogramFile wraps a .tgm file on disk and provides:
//!   create()         — new file (truncates if exists)
//!   open()           — existing file for reading
//!   append()         — encode and append a message
//!   message_count()  — how many valid messages are in the file
//!   read_message()   — raw bytes of message\[i\]
//!   messages()       — all raw message bytes as a `Vec<Vec<u8>>`
//!   decode_message() — decode message\[i\] → `(GlobalMetadata, Vec<(DataObjectDescriptor, Vec<u8>)>)`
//!   path()           — the file path
//!
//! The file is lazily scanned: no I/O happens on open()/create().
//! The scan runs on the first call that needs the message list.

use std::collections::BTreeMap;

use ciborium::Value;
use tensogram::{
    ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions, GlobalMetadata,
    TensogramFile, decode_metadata, scan,
};

fn make_forecast_message(
    param: &str,
    step: i64,
) -> (GlobalMetadata, DataObjectDescriptor, Vec<u8>) {
    let mars = Value::Map(vec![
        (Value::Text("param".into()), Value::Text(param.into())),
        (Value::Text("step".into()), Value::Integer(step.into())),
        (Value::Text("date".into()), Value::Text("20260401".into())),
        (Value::Text("type".into()), Value::Text("fc".into())),
    ]);
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
        shape: vec![721u64, 1440],
        strides: vec![1440u64, 1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
        hash: None,
    };

    // Payload: zeros stand in for real forecast data
    let data = vec![0u8; 721 * 1440 * 4];
    (global_meta, desc, data)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("forecast.tgm");

    // ── 1. Create and append messages ─────────────────────────────────────────
    {
        let mut file = TensogramFile::create(&path)?;
        println!("Created: {}", file.source());

        let params_steps = [
            ("2t", 0i64),
            ("10u", 0),
            ("10v", 0),
            ("msl", 0),
            ("2t", 6),
            ("10u", 6),
            ("10v", 6),
            ("msl", 6),
        ];

        for (param, step) in params_steps {
            let (meta, desc, data) = make_forecast_message(param, step);
            file.append(&meta, &[(&desc, &data)], &EncodeOptions::default())?;
        }

        println!("Appended {} messages", params_steps.len());
        // message_count() triggers the lazy scan on first call
        println!("message_count() = {}", file.message_count()?);
    }

    // ── 2. Open and inspect ───────────────────────────────────────────────────
    {
        let file = TensogramFile::open(&path)?;
        // No I/O yet. Scan runs on first access below.

        let count = file.message_count()?; // ← scan happens here
        println!("\nOpened: {} messages", count);
        assert_eq!(count, 8);

        // ── 3. Decode by index ────────────────────────────────────────────────
        //
        // After the scan, every access is a seek + read — no re-scan.
        println!("\nRandom access by index:");
        for i in [0, 3, 7] {
            let (meta, objects) = file.decode_message(i, &DecodeOptions::default())?;
            let mars = meta.base.first().and_then(|e| e.get("mars")).unwrap();
            let param = get_text(mars, "param");
            let step = get_int(mars, "step");
            println!(
                "  [{i}] param={param:5}  step={step:2}  data={} bytes",
                objects[0].1.len()
            );
        }

        // ── 4. Read raw bytes ─────────────────────────────────────────────────
        //
        // read_message() returns the complete wire-format bytes.
        // Useful when you want to forward a message unchanged.
        let raw = file.read_message(0)?;
        println!(
            "\nread_message(0): {} bytes  magic={:?}",
            raw.len(),
            std::str::from_utf8(&raw[0..8]).unwrap()
        );

        // ── 5. messages() — all as Vec<Vec<u8>> ───────────────────────────────
        //
        // Reads all messages into memory. Convenient for small files.
        #[allow(deprecated)]
        let all = file.messages()?;
        println!("\nmessages(): {} raw buffers loaded into memory", all.len());
        for (i, msg) in all.iter().enumerate() {
            let meta = decode_metadata(msg)?;
            let mars = meta.base.first().and_then(|e| e.get("mars")).unwrap();
            let step = get_int(mars, "step");
            let param = get_text(mars, "param");
            println!("  [{i}] param={param:5}  step={step}");
        }
    }

    // ── 6. Scan a raw file buffer ─────────────────────────────────────────────
    //
    // scan() is the lower-level primitive that TensogramFile uses internally.
    // Call it directly when you have an in-memory buffer (e.g. from a socket).
    {
        let file_bytes = std::fs::read(&path)?;
        let offsets = scan(&file_bytes);
        println!(
            "\nscan() on raw file bytes: {} messages found",
            offsets.len()
        );
    }

    println!("\nFile API example complete.");
    Ok(())
}

// ── helpers to navigate MARS map ──────────────────────────────────────────────

fn get_text<'a>(map: &'a Value, key: &str) -> &'a str {
    if let Value::Map(entries) = map {
        for (k, v) in entries {
            if matches!(k, Value::Text(s) if s == key)
                && let Value::Text(t) = v
            {
                return t;
            }
        }
    }
    "?"
}

fn get_int(map: &Value, key: &str) -> i64 {
    if let Value::Map(entries) = map {
        for (k, v) in entries {
            if matches!(k, Value::Text(s) if s == key)
                && let Value::Integer(i) = v
            {
                let n: i128 = (*i).into();
                return n as i64;
            }
        }
    }
    -1
}
