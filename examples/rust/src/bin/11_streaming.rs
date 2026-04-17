// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Example 11 — Streaming encoder
//!
//! The StreamingEncoder writes Tensogram frames progressively to any
//! `std::io::Write` sink. This enables encoding to a file, socket, or pipe
//! without buffering the entire message in memory.
//!
//! The preamble is written with total_length=0 (streaming mode), and
//! index + hash frames are written as footers when `finish()` is called.

use std::collections::BTreeMap;
use std::io::BufWriter;

use tensogram_core::streaming::StreamingEncoder;
use tensogram_core::{
    ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions, GlobalMetadata, decode,
};

fn make_descriptor(shape: Vec<u64>) -> DataObjectDescriptor {
    let ndim = shape.len() as u64;
    let mut strides = vec![0u64; shape.len()];
    if !shape.is_empty() {
        strides[shape.len() - 1] = 1;
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim,
        shape,
        strides,
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Stream to an in-memory buffer ────────────────────────────────────
    let meta = GlobalMetadata {
        version: 2,
        extra: {
            let mut m = BTreeMap::new();
            m.insert(
                "centre".to_string(),
                ciborium::Value::Text("ecmwf".to_string()),
            );
            m
        },
        ..Default::default()
    };

    let buffer = Vec::new();
    let writer = BufWriter::new(buffer);
    let mut encoder = StreamingEncoder::new(writer, &meta, &EncodeOptions::default())?;

    // Write 3 objects progressively
    for i in 0..3 {
        let desc = make_descriptor(vec![4]);
        let data: Vec<u8> = (0..4u32)
            .map(|j| (i * 4 + j) as f32)
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        encoder.write_object(&desc, &data)?;
        println!(
            "  Wrote object {i}: {} bytes total",
            encoder.bytes_written()
        );
    }

    let writer = encoder.finish()?;
    let message = writer.into_inner()?;
    println!("Streaming encode complete: {} bytes", message.len());

    // ── 2. Decode the streamed message ──────────────────────────────────────
    let (decoded_meta, objects) = decode(
        &message,
        &DecodeOptions {
            verify_hash: true,
            ..Default::default()
        },
    )?;
    println!(
        "\nDecoded: version={}, {} objects",
        decoded_meta.version,
        objects.len()
    );
    println!("  Base metadata: {:?}", decoded_meta.base);

    for (i, (desc, data)) in objects.iter().enumerate() {
        let values: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        println!("  Object {i}: shape={:?} values={values:?}", desc.shape);
    }

    // ── 3. Stream to a temporary file ───────────────────────────────────────
    let tmp = tempfile::NamedTempFile::new()?;
    println!("\nStreaming to file: {:?}", tmp.path());

    let file_writer = BufWriter::new(tmp.reopen()?);
    let mut encoder = StreamingEncoder::new(file_writer, &meta, &EncodeOptions::default())?;

    let desc = make_descriptor(vec![10, 10]);
    let data = vec![0u8; 10 * 10 * 4];
    encoder.write_object(&desc, &data)?;
    encoder.finish()?;

    // Read it back
    let file_bytes = std::fs::read(tmp.path())?;
    let (_, objects) = decode(&file_bytes, &DecodeOptions::default())?;
    println!(
        "Read back from file: {} object(s), shape={:?}",
        objects.len(),
        objects[0].0.shape
    );

    Ok(())
}
