// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Example 19 — Message metadata known only at the end of a stream
//!
//! A streaming producer often cannot describe its own message up front: which
//! of the requested parameters actually had data, the observed value range,
//! how many objects ended up written.  `set_footer_metadata` records those
//! facts while encoding; they are merged into the footer metadata frame, which
//! supersedes the header frame on decode.
//!
//! The header can therefore carry a provisional value and the footer the final
//! one — footer `_extra_` keys win on conflict.

use std::collections::BTreeMap;

use ciborium::Value;
use tensogram::streaming::StreamingEncoder;
use tensogram::{
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
        masks: None,
    }
}

/// The parameters a consumer asked for.  Only some of them turn out to have
/// data — which ones is not known until the last field has been fetched.
const REQUESTED: [&str; 3] = ["2t", "10u", "tp"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The header metadata states the request and marks the result
    // provisional: at this point the producer does not yet know what it will
    // be able to write.
    let header_meta = GlobalMetadata {
        extra: BTreeMap::from([
            (
                "requested_params".to_string(),
                Value::Array(
                    REQUESTED
                        .iter()
                        .map(|p| Value::Text(p.to_string()))
                        .collect(),
                ),
            ),
            ("status".to_string(), Value::Text("provisional".to_string())),
        ]),
        ..Default::default()
    };

    let mut encoder = StreamingEncoder::new(Vec::new(), &header_meta, &EncodeOptions::default())?;

    // Stream one object per parameter that has data.  "tp" has none, so no
    // object is written for it.
    let mut written: Vec<Value> = Vec::new();
    let mut maximum = f32::MIN;
    for (i, param) in REQUESTED.iter().enumerate() {
        if *param == "tp" {
            println!("  {param}: no data — skipped");
            continue;
        }
        let values: Vec<f32> = (0..4).map(|j| 250.0 + (i * 4 + j) as f32).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

        // Per-object metadata goes in a preceder frame, as usual.
        let mut preceder = BTreeMap::new();
        preceder.insert("param".to_string(), Value::Text((*param).to_string()));
        encoder.write_preceder(preceder)?;
        encoder.write_object(&make_descriptor(vec![4]), &data)?;

        written.push(Value::Text((*param).to_string()));
        maximum = maximum.max(values.iter().copied().fold(f32::MIN, f32::max));
        println!("  {param}: wrote 4 values");
    }

    // Now the message-level facts are known.  Record them for the footer
    // frame; `status` replaces the provisional header value.
    encoder.set_footer_metadata(BTreeMap::from([
        ("status".to_string(), Value::Text("complete".to_string())),
        ("written_params".to_string(), Value::Array(written)),
        ("value_maximum".to_string(), Value::Float(maximum.into())),
    ]));

    // Repeated calls accumulate, so late additions need no bookkeeping of
    // their own.
    encoder.set_footer_metadata(BTreeMap::from([(
        "producer".to_string(),
        Value::Text("footer-metadata-example".to_string()),
    )]));

    let message = encoder.finish()?;
    println!("\nStreamed {} bytes", message.len());

    // ── Decoding sees the final values ──────────────────────────────────────
    let (meta, objects) = decode(&message, &DecodeOptions::default())?;
    println!("Decoded {} object(s)", objects.len());
    println!("  status:         {:?}", meta.extra.get("status"));
    println!("  written_params: {:?}", meta.extra.get("written_params"));
    println!("  value_maximum:  {:?}", meta.extra.get("value_maximum"));
    println!(
        "  requested_params (from the header frame): {:?}",
        meta.extra.get("requested_params")
    );

    // A reader that parses only the header metadata frame — a forward scan
    // that never reaches the footer — keeps seeing the provisional value.
    let header_only = tensogram::decode_metadata(&message)?;
    println!(
        "\nHeader-only read still reports status={:?}",
        header_only.extra.get("status")
    );

    Ok(())
}
