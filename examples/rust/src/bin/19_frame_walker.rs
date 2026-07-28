// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Example 19 — Frame introspection: walking a message's structure
//!
//! Every .tgm message is a 24-byte preamble, a sequence of frames, and a
//! postamble. `message_header()` reads the envelope and `frames()` walks the
//! frames — both *without* decoding payloads or CBOR, so they are cheap enough
//! to run over a whole file.
//!
//! This example:
//!   - Reads the envelope (wire version, total length, structural flags)
//!   - Walks the frames of a buffered (random-access) message
//!   - Shows that `offset`/`length` tile the message and `payload` is the
//!     frame *content* (frame header and type-specific footer stripped)
//!   - Contrasts a streaming message, whose index/hashes live in the footer
//!   - Uses the walker to *prove* where `aggregate_hash` puts its hash frames
//!
//! Note: `frames()` takes ONE message. For a multi-message buffer, use `scan()`
//! to get each `(offset, length)` and slice first — see example 07.

use std::collections::BTreeMap;

use tensogram::{
    ByteOrder, DataObjectDescriptor, Dtype, EncodeOptions, FrameInfo, GlobalMetadata,
    StreamingEncoder, encode, frames, message_header,
};

fn descriptor(shape: Vec<u64>) -> DataObjectDescriptor {
    let strides = {
        let mut s = vec![1u64; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * shape[i + 1];
        }
        s
    };
    DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: shape.len() as u64,
        shape,
        strides,
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    }
}

/// Print the envelope and the frame table for one message.
fn describe(label: &str, msg: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let header = message_header(msg)?;
    println!("\n=== {label} ({} bytes) ===", msg.len());
    println!(
        "  wire version {}  total_length {}{}",
        header.version,
        header.total_length,
        if header.total_length == 0 {
            "  (streaming: never back-filled)"
        } else {
            ""
        }
    );
    // `has_header_index` is the reliable random-access discriminator: the
    // streaming encoder also writes a HeaderMetadata frame, so
    // `has_header_metadata` is true in both modes.
    println!(
        "  layout: {}",
        if header.has_header_index() {
            "random-access (index in the header)"
        } else {
            "streaming (index in the footer)"
        }
    );
    println!(
        "  metadata hdr/ftr {}/{}   index hdr/ftr {}/{}   hashes hdr/ftr {}/{}   preceder {}",
        header.has_header_metadata(),
        header.has_footer_metadata(),
        header.has_header_index(),
        header.has_footer_index(),
        header.has_header_hashes(),
        header.has_footer_hashes(),
        header.has_preceder_metadata(),
    );

    println!(
        "  {:<20} {:>8} {:>8} {:>9}  hash",
        "frame type", "offset", "length", "payload"
    );
    let mut covered = 0usize;
    for frame in frames(msg)? {
        let f: FrameInfo = frame?;
        println!(
            "  {:<20} {:>8} {:>8} {:>9}  {}",
            format!("{:?}", f.frame_type),
            f.offset,
            f.length,
            f.payload.len(),
            if f.has_hash() { "yes" } else { "no" }
        );
        covered += f.length;
    }
    println!(
        "  frames cover {covered} of {} bytes (rest: envelope + padding)",
        msg.len()
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let meta = GlobalMetadata::default();
    let desc = descriptor(vec![4]);
    let data = vec![0u8; 16];

    // ── Buffered encode: the encoder knows everything up front, so the
    //    metadata/index/hashes go in the HEADER (random access).
    let buffered = encode(
        &meta,
        &[(&desc, &data), (&desc, &data)],
        &EncodeOptions::default(),
    )?;
    describe("buffered (random-access), 2 objects", &buffered)?;

    // ── Streaming encode: the preamble is written before any object, so the
    //    index/hashes can only go in the FOOTER.
    let streamed = {
        let mut enc = StreamingEncoder::new(
            std::io::Cursor::new(Vec::new()),
            &meta,
            &EncodeOptions::default(),
        )?;
        enc.write_object(&desc, &data)?;
        enc.finish()?.into_inner()
    };
    describe("streaming, 1 object", &streamed)?;

    // ── The walker is the honest way to verify where hashes actually landed.
    println!("\n=== aggregate_hash placement (verified by walking) ===");
    for policy in [
        tensogram::AggregateHashPolicy::Header,
        tensogram::AggregateHashPolicy::Footer,
        tensogram::AggregateHashPolicy::Both,
        tensogram::AggregateHashPolicy::None,
    ] {
        let opts = EncodeOptions {
            aggregate_hash: policy,
            ..Default::default()
        };
        let msg = encode(&meta, &[(&desc, &data)], &opts)?;
        let mut header_hash = false;
        let mut footer_hash = false;
        for frame in frames(&msg)? {
            match frame?.frame_type {
                tensogram::FrameType::HeaderHash => header_hash = true,
                tensogram::FrameType::FooterHash => footer_hash = true,
                _ => {}
            }
        }
        println!(
            "  {:<8} -> header hash: {header_hash:<5}  footer hash: {footer_hash}",
            format!("{policy:?}")
        );
    }

    println!("\nDone.");
    Ok(())
}
