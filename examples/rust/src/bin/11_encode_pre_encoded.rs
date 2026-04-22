// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Example 11 — encode_pre_encoded round-trip
//!
//! Demonstrates the GPU-pipeline pattern: a pre-packed buffer is produced
//! outside the library, its descriptor declares the applied
//! encoding/compression, and the bytes are handed directly to
//! [`encode_pre_encoded`] (and [`StreamingEncoder::write_object_pre_encoded`])
//! so the library wraps them into a wire-format message without re-running
//! the pipeline.
//!
//! Covers:
//!   1. simple_packing + szip round-trip
//!   2. encoding="none" raw round-trip
//!   3. [`StreamingEncoder`] variant combining both objects into one message
//!
//! IMPORTANT — bit-vs-byte gotcha:
//!   `szip_block_offsets` are **BIT** offsets, not byte offsets.  For
//!   example, if a compressed block starts at byte 16 the offset is 128
//!   (= 16 * 8).  Getting this wrong silently breaks `decode_range()`.

use std::collections::BTreeMap;

use ciborium::Value;
use tensogram::{
    ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions, GlobalMetadata,
    StreamingEncoder, decode, encode_pre_encoded,
};
use tensogram_encodings::pipeline::{CompressionType, EncodingType, FilterType, PipelineConfig};
use tensogram_encodings::{pipeline, simple_packing};

fn plain_desc(shape: u64, dtype: Dtype, byte_order: ByteOrder) -> DataObjectDescriptor {
    DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![shape],
        strides: vec![1],
        dtype,
        byte_order,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    }
}

fn decode_f64_be(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|chunk| {
            let arr: [u8; 8] = chunk
                .try_into()
                .expect("chunks_exact(8) always yields 8-byte slices");
            f64::from_be_bytes(arr)
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Source data: a 1-D f64 signal ─────────────────────────────────────
    let n = 1024usize;
    let values: Vec<f64> = (0..n).map(|i| 250.0 + (i as f64) * 0.25).collect();
    let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
    println!("Source: {n} f64 values  raw={} bytes", raw_bytes.len());

    // ── 2. Compute simple_packing parameters and build descriptor ────────────
    let bits_per_value = 16u32;
    let packing = simple_packing::compute_params(&values, bits_per_value, 0)?;
    println!(
        "Packing params: ref={:.4}  bsf={}  dsf={}  bpv={}",
        packing.reference_value,
        packing.binary_scale_factor,
        packing.decimal_scale_factor,
        packing.bits_per_value,
    );

    let mut params = BTreeMap::new();
    params.insert(
        "reference_value".to_string(),
        Value::Float(packing.reference_value),
    );
    params.insert(
        "binary_scale_factor".to_string(),
        Value::Integer((packing.binary_scale_factor as i64).into()),
    );
    params.insert(
        "decimal_scale_factor".to_string(),
        Value::Integer((packing.decimal_scale_factor as i64).into()),
    );
    params.insert(
        "bits_per_value".to_string(),
        Value::Integer((packing.bits_per_value as i64).into()),
    );
    params.insert("szip_rsi".to_string(), Value::Integer(128.into()));
    params.insert("szip_block_size".to_string(), Value::Integer(16.into()));
    params.insert("szip_flags".to_string(), Value::Integer(8_i64.into()));

    let mut packed_desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![n as u64],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Big,
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "szip".to_string(),
        params,
        masks: None,
    };

    // ── 3. Produce the pre-packed bytes ──────────────────────────────────────
    //
    // A real caller would produce these on a GPU or in an HPC framework.
    // For self-containment the example drives the library's own forward
    // pipeline to obtain bytes byte-for-byte identical to what an
    // external kernel would emit.
    let config = PipelineConfig {
        encoding: EncodingType::SimplePacking(packing.clone()),
        filter: FilterType::None,
        compression: CompressionType::Szip {
            rsi: 128,
            block_size: 16,
            flags: 8,
            bits_per_sample: bits_per_value,
        },
        num_values: n,
        byte_order: ByteOrder::Big,
        dtype_byte_width: 8,
        swap_unit_size: 8,
        compression_backend: Default::default(),
        intra_codec_threads: 0,
        compute_hash: false,
    };
    let packed = pipeline::encode_pipeline(&raw_bytes, &config)?;
    let packed_bytes = packed.encoded_bytes;
    println!(
        "Packed payload: {} bytes  ({} bits x {} values)",
        packed_bytes.len(),
        bits_per_value,
        n,
    );

    // Attach szip's per-block bit offsets so the descriptor supports
    // partial-range decode.  `encode()` does this automatically from
    // the pipeline result; `encode_pre_encoded()` relies on the caller
    // to carry the offsets forward.
    if let Some(offsets) = packed.block_offsets {
        packed_desc.params.insert(
            "szip_block_offsets".to_string(),
            Value::Array(
                offsets
                    .iter()
                    .map(|&o| Value::Integer((o as i64).into()))
                    .collect(),
            ),
        );
    }

    // ── 4. Wrap the pre-encoded bytes into a wire-format message ─────────────
    let meta = GlobalMetadata::default();
    let message = encode_pre_encoded(
        &meta,
        &[(&packed_desc, packed_bytes.as_slice())],
        &EncodeOptions::default(),
    )?;
    println!("Wire message: {} bytes", message.len());

    // ── 5. Decode and verify ────────────────────────────────────────────────
    let (decoded_meta, decoded_objects) = decode(
        &message,
        &DecodeOptions {
            native_byte_order: false,
            ..DecodeOptions::default()
        },
    )?;
    assert_eq!(decoded_meta.version, 3);
    let (decoded_desc, decoded_payload) = decoded_objects
        .first()
        .ok_or_else(|| std::io::Error::other("missing decoded object"))?;
    assert_eq!(decoded_desc.encoding, "simple_packing");
    assert_eq!(decoded_desc.compression, "szip");

    let decoded_values = decode_f64_be(decoded_payload);
    assert_eq!(decoded_values.len(), n);
    let max_err = values
        .iter()
        .zip(decoded_values.iter())
        .map(|(v, d)| (v - d).abs())
        .fold(0.0f64, f64::max);
    println!("Max quantisation error: {max_err:.6}");
    assert!(max_err < 0.01, "error {max_err} exceeds tolerance");

    // ── 6. encoding="none" variant ──────────────────────────────────────────
    //
    // The simplest case: raw payload bytes, no encoding applied.
    let raw_values: Vec<f32> = (0..50).map(|i| i as f32).collect();
    let raw_data: Vec<u8> = raw_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let raw_desc = plain_desc(raw_values.len() as u64, Dtype::Float32, ByteOrder::Little);
    let raw_msg = encode_pre_encoded(
        &meta,
        &[(&raw_desc, raw_data.as_slice())],
        &EncodeOptions::default(),
    )?;
    // Decode with native_byte_order = false so the returned bytes are
    // in the wire byte-order — identical to `raw_data` on any host.
    let (_, raw_objs) = decode(
        &raw_msg,
        &DecodeOptions {
            native_byte_order: false,
            ..DecodeOptions::default()
        },
    )?;
    assert_eq!(raw_objs[0].1, raw_data);
    println!("Raw encoding=none round-trip: OK");

    // ── 7. StreamingEncoder variant ─────────────────────────────────────────
    //
    // `write_object_pre_encoded` accepts the same pre-packed bytes the
    // buffered API above did, so a single streaming message can carry a
    // mix of fully-encoded and raw objects.
    let buf = Vec::new();
    let mut enc = StreamingEncoder::new(buf, &meta, &EncodeOptions::default())?;
    enc.write_object_pre_encoded(&packed_desc, packed_bytes.as_slice())?;
    enc.write_object_pre_encoded(&raw_desc, raw_data.as_slice())?;
    let stream_msg = enc.finish()?;

    let (_, stream_objs) = decode(
        &stream_msg,
        &DecodeOptions {
            native_byte_order: false,
            ..DecodeOptions::default()
        },
    )?;
    assert_eq!(stream_objs.len(), 2);
    let stream_values = decode_f64_be(&stream_objs[0].1);
    let stream_max_err = values
        .iter()
        .zip(stream_values.iter())
        .map(|(v, d)| (v - d).abs())
        .fold(0.0f64, f64::max);
    assert!(stream_max_err < 0.01);
    assert_eq!(stream_objs[1].1, raw_data);
    println!("Streaming pre-encoded: OK");

    println!(
        "\nOK: all pre-encoded round-trips succeeded ({} bytes)",
        message.len()
    );
    Ok(())
}
