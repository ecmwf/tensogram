// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! 12 — convert-netcdf via the library API
//!
//! End-to-end example: call `tensogram_netcdf::convert_netcdf_file()` directly
//! on a fixture file shipped with the `tensogram-netcdf` crate, write the
//! resulting messages to a temporary file, and read them back through
//! `TensogramFile`.
//!
//! Requires libnetcdf at the OS level. Build and run:
//! ```bash
//! brew install netcdf            # macOS
//! apt install libnetcdf-dev      # Debian/Ubuntu
//! cargo run -p tensogram-rust-examples --bin 12_convert_netcdf --features netcdf
//! ```
//!
//! The example is gated behind the `netcdf` feature so contributors who
//! don't have libnetcdf installed can still build the rest of the rust
//! examples without errors.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use tempfile::tempdir;
use tensogram::{TensogramFile, decode_metadata};
use tensogram_netcdf::{ConvertOptions, DataPipeline, SplitBy, convert_netcdf_file};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Locate a fixture from the tensogram-netcdf crate ──────────────
    //
    // The example crate lives at examples/rust, so we walk up two levels
    // and dive into rust/tensogram-netcdf/testdata to find the shipped
    // CF-compliant temperature file.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture = manifest_dir
        .join("..")
        .join("..")
        .join("rust")
        .join("tensogram-netcdf")
        .join("testdata")
        .join("cf_temperature.nc");
    let fixture = fixture.canonicalize()?;
    println!("1. Using fixture: {}", fixture.display());

    // ── 2. Configure the conversion ─────────────────────────────────────
    let options = ConvertOptions {
        // One Tensogram message per NetCDF variable.
        split_by: SplitBy::Variable,
        // Lift CF metadata into a stable allow-list under base[i]["cf"].
        cf: true,
        // Encode float64 variables with 24-bit simple_packing and zstd
        // compression. Non-f64 variables (the f32 lat/lon coordinates)
        // are skipped for the encoding stage with a stderr warning, but
        // still get zstd compression.
        pipeline: DataPipeline {
            encoding: "simple_packing".to_string(),
            bits: Some(24),
            compression: "zstd".to_string(),
            compression_level: Some(3),
            ..Default::default()
        },
        ..Default::default()
    };
    println!(
        "2. Options: split_by={:?}, cf={}, encoding={}, compression={}",
        options.split_by, options.cf, options.pipeline.encoding, options.pipeline.compression
    );

    // ── 3. Run the conversion ───────────────────────────────────────────
    let messages = convert_netcdf_file(&fixture, &options)?;
    println!("3. Produced {} Tensogram message(s)", messages.len());

    // ── 4. Write to a temporary .tgm file ───────────────────────────────
    let dir = tempdir()?;
    let out_path = dir.path().join("demo.tgm");
    let mut out = File::create(&out_path)?;
    for msg in &messages {
        out.write_all(msg)?;
    }
    drop(out);
    println!("4. Wrote {}", out_path.display());

    // ── 5. Read it back with TensogramFile ──────────────────────────────
    let tgm = TensogramFile::open(&out_path)?;
    let count = tgm.message_count()?;
    println!("5. message_count = {count}");

    for i in 0..count {
        let raw = tgm.read_message(i)?;
        let meta = decode_metadata(&raw)?;
        let entry = meta.base.first();
        let name = entry.and_then(|e| e.get("name")).and_then(|v| {
            if let ciborium::Value::Text(s) = v {
                Some(s.as_str())
            } else {
                None
            }
        });
        let has_cf = entry.is_some_and(|e| e.contains_key("cf"));
        println!(
            "   message[{i}] name={name:?} cf_lifted={has_cf}",
            name = name.unwrap_or("?")
        );
    }

    println!("\nOK");
    Ok(())
}
