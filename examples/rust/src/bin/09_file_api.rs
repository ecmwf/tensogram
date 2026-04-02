//! Example 09 — File API (TensogramFile)
//!
//! TensogramFile wraps a .tgm file on disk and provides:
//!   create()         — new file (truncates if exists)
//!   open()           — existing file for reading
//!   append()         — encode and append a message
//!   message_count()  — how many valid messages are in the file
//!   read_message()   — raw bytes of message\[i\]
//!   messages()       — all raw message bytes as a `Vec<Vec<u8>>`
//!   decode_message() — decode message\[i\] → (Metadata, `Vec<Vec<u8>>`)
//!   path()           — the file path
//!
//! The file is lazily scanned: no I/O happens on open()/create().
//! The scan runs on the first call that needs the message list.

use std::collections::BTreeMap;

use ciborium::Value;
use tensogram_core::{
    decode_metadata, scan, ByteOrder, DecodeOptions, Dtype, EncodeOptions, Metadata,
    ObjectDescriptor, PayloadDescriptor, TensogramFile,
};

fn make_forecast_message(param: &str, step: i64) -> (Metadata, Vec<u8>) {
    let mars = Value::Map(vec![
        (Value::Text("param".into()), Value::Text(param.into())),
        (Value::Text("step".into()), Value::Integer(step.into())),
        (Value::Text("date".into()), Value::Text("20260401".into())),
        (Value::Text("type".into()), Value::Text("fc".into())),
    ]);
    let mut extra = BTreeMap::new();
    extra.insert("mars".to_string(), mars);

    let shape = vec![721u64, 1440];
    let strides = vec![1440u64, 1];
    let metadata = Metadata {
        version: 1,
        objects: vec![ObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 2,
            shape,
            strides,
            dtype: Dtype::Float32,
            extra: BTreeMap::new(),
        }],
        payload: vec![PayloadDescriptor {
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        }],
        extra,
    };

    // Payload: zeros stand in for real forecast data
    let data = vec![0u8; 721 * 1440 * 4];
    (metadata, data)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("forecast.tgm");

    // ── 1. Create and append messages ─────────────────────────────────────────
    {
        let mut file = TensogramFile::create(&path)?;
        println!("Created: {}", file.path().display());

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
            let (meta, data) = make_forecast_message(param, step);
            file.append(&meta, &[&data], &EncodeOptions::default())?;
        }

        println!("Appended {} messages", params_steps.len());
        // message_count() triggers the lazy scan on first call
        println!("message_count() = {}", file.message_count()?);
    }

    // ── 2. Open and inspect ───────────────────────────────────────────────────
    {
        let mut file = TensogramFile::open(&path)?;
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
            let mars = &meta.extra["mars"];
            let param = get_text(mars, "param");
            let step = get_int(mars, "step");
            println!(
                "  [{i}] param={param:5}  step={step:2}  data={} bytes",
                objects[0].len()
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
        let all = file.messages()?;
        println!("\nmessages(): {} raw buffers loaded into memory", all.len());
        for (i, msg) in all.iter().enumerate() {
            let meta = decode_metadata(msg)?;
            let step = get_int(&meta.extra["mars"], "step");
            let param = get_text(&meta.extra["mars"], "param");
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
            if matches!(k, Value::Text(s) if s == key) {
                if let Value::Text(t) = v {
                    return t;
                }
            }
        }
    }
    "?"
}

fn get_int(map: &Value, key: &str) -> i64 {
    if let Value::Map(entries) = map {
        for (k, v) in entries {
            if matches!(k, Value::Text(s) if s == key) {
                if let Value::Integer(i) = v {
                    let n: i128 = (*i).into();
                    return n as i64;
                }
            }
        }
    }
    -1
}
