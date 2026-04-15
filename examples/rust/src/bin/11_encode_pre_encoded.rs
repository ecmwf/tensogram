// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Example 11 — encode_pre_encoded round-trip
//!
//! Demonstrates the GPU-pipeline pattern: a pre-packed buffer is produced first,
//! its descriptor declares the applied encoding/compression, and then the bytes
//! are handed directly to `encode_pre_encoded()`.
//!
//! IMPORTANT: `szip_block_offsets` are BIT offsets, not byte offsets.

use std::collections::BTreeMap;

use ciborium::Value;
use tensogram_core::{
    decode, encode_pre_encoded, ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype,
    EncodeOptions, GlobalMetadata,
};
use tensogram_encodings::pipeline::{CompressionType, EncodingType, FilterType, PipelineConfig};
use tensogram_encodings::{pipeline, simple_packing};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Source data: a simple 1D float64 signal.
    let values: Vec<f64> = (0..1024).map(|i| 250.0 + (i as f64) * 0.25).collect();
    let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();

    // 2. Build descriptor params for simple_packing + szip.
    let bits_per_value = 16u32;
    let packing = simple_packing::compute_params(&values, bits_per_value, 0)?;

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

    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![values.len() as u64],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Big,
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "szip".to_string(),
        params,
        hash: None,
    };

    // 3. Fake the GPU pipeline by running the internal forward pipeline once.
    //    This gives us realistic pre-packed bytes without using any external GPU code.
    let config = PipelineConfig {
        encoding: EncodingType::SimplePacking(packing.clone()),
        filter: FilterType::None,
        compression: CompressionType::Szip {
            rsi: 128,
            block_size: 16,
            flags: 8,
            bits_per_sample: bits_per_value,
        },
        num_values: values.len(),
        byte_order: ByteOrder::Big,
        dtype_byte_width: 8,
        swap_unit_size: 8, // f64
        compression_backend: Default::default(),
    };
    let packed = pipeline::encode_pipeline(&raw_bytes, &config)?;
    let packed_bytes = packed.encoded_bytes;
    let packed_desc = desc.clone();
    let _ = packed.block_offsets;

    // 4. Feed the pre-encoded bytes directly to the API under test.
    let meta = GlobalMetadata::default();
    let options = EncodeOptions::default();
    let message = encode_pre_encoded(&meta, &[(&packed_desc, packed_bytes.as_slice())], &options)?;

    // 5. Decode the message and verify the round-trip.
    let (decoded_meta, decoded_objects) = decode(
        &message,
        &DecodeOptions {
            native_byte_order: false,
            ..DecodeOptions::default()
        },
    )?;
    let (decoded_desc, decoded_payload) = decoded_objects
        .first()
        .ok_or_else(|| std::io::Error::other("missing decoded object"))?;

    assert_eq!(decoded_meta.version, 2);
    assert_eq!(decoded_desc.encoding, "simple_packing");
    assert_eq!(decoded_desc.compression, "szip");
    assert_eq!(
        decoded_payload,
        raw_bytes.as_slice(),
        "round-trip must recover the source bytes"
    );

    println!(
        "OK: encode_pre_encoded round-trip succeeded ({} bytes)",
        message.len()
    );
    Ok(())
}
