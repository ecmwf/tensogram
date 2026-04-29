// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Benchmark: `encode_full_pipeline` vs `encode_pre_encoded`
//!
//! Measures the cost of:
//! - Full pipeline: simple_packing 24-bit → szip (encoding + compression + framing)
//! - Pre-encoded path: framing only (no encoding/compression, caller bytes passed through)
//!
//! Setup: 16M float64 values (~128 MiB raw), sinusoidal synthetic data.
//! The pre-encoded bytes are captured once outside the timed loop to simulate
//! "the GPU produced these bytes earlier" — they are then fed directly to
//! `encode_pre_encoded`, which skips simple_packing + szip entirely.

use std::collections::BTreeMap;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use tensogram::{
    ByteOrder, DataObjectDescriptor, Dtype, EncodeOptions, GlobalMetadata, encode,
    encode_pre_encoded, framing,
};
use tensogram_encodings::simple_packing::compute_params;

const N_FLOATS: usize = 16 * 1024 * 1024; // 16 Mi float64 (~128 MiB raw)

fn bench_encode_paths(c: &mut Criterion) {
    // ── Synthesize 16M float64 values ────────────────────────────────────────
    // Sinusoidal pattern with a 280-unit offset; smooth fields compress well with szip.
    let raw: Vec<f64> = (0..N_FLOATS)
        .map(|i| (i as f64 * 0.0001).sin() * 1000.0 + 280.0)
        .collect();

    // Encode as big-endian bytes (matches ByteOrder::Big descriptor below).
    let raw_bytes: Vec<u8> = raw.iter().flat_map(|v| v.to_be_bytes()).collect();

    // ── Compute simple_packing params from the actual data ────────────────────
    let sp_params = compute_params(&raw, 24, 0).expect("compute_params failed");

    // ── Build params map (ciborium Values) ────────────────────────────────────
    let mut params_map: BTreeMap<String, ciborium::Value> = BTreeMap::new();
    params_map.insert(
        "sp_reference_value".to_string(),
        ciborium::Value::Float(sp_params.reference_value),
    );
    params_map.insert(
        "sp_binary_scale_factor".to_string(),
        ciborium::Value::Integer((sp_params.binary_scale_factor as i64).into()),
    );
    params_map.insert(
        "sp_decimal_scale_factor".to_string(),
        ciborium::Value::Integer((sp_params.decimal_scale_factor as i64).into()),
    );
    params_map.insert(
        "sp_bits_per_value".to_string(),
        ciborium::Value::Integer((sp_params.bits_per_value as i64).into()),
    );
    // szip parameters (RSI=128 restart interval, block_size=16, flags=8 = AEC_DATA_PREPROCESS)
    params_map.insert(
        "szip_rsi".to_string(),
        ciborium::Value::Integer(128_i64.into()),
    );
    params_map.insert(
        "szip_block_size".to_string(),
        ciborium::Value::Integer(16_i64.into()),
    );
    params_map.insert(
        "szip_flags".to_string(),
        ciborium::Value::Integer(8_i64.into()),
    );

    // ── Build the DataObjectDescriptor ───────────────────────────────────────
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        shape: vec![N_FLOATS as u64],
        strides: vec![1],
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Big,
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "szip".to_string(),
        params: params_map,
        masks: None,
    };

    let meta = GlobalMetadata::default();
    // Disable hashing to isolate encode pipeline cost (not hash cost).
    let opts = EncodeOptions {
        hashing: false,
        ..Default::default()
    };

    // ── Prime: encode once outside the timed loop ─────────────────────────────
    // This simulates "the GPU already produced these bytes".
    // We capture the encoded payload and updated descriptor (with szip_block_offsets)
    // by calling encode() and then extracting via framing::decode_message.
    let wire_msg = encode(&meta, &[(&desc, &raw_bytes)], &opts).expect("priming encode failed");
    let gpu_desc: DataObjectDescriptor;
    let gpu_bytes: Vec<u8>;
    {
        let decoded_msg =
            framing::decode_message(&wire_msg).expect("priming framing decode failed");
        let (pre_desc, pre_bytes_slice, _mask_region, _) = &decoded_msg.objects[0];
        gpu_desc = pre_desc.clone();
        gpu_bytes = pre_bytes_slice.to_vec();
    }

    // ── Benchmark group ───────────────────────────────────────────────────────
    let mut group = c.benchmark_group("encode_pre_encoded_vs_full");
    // Report throughput as raw bytes in (128 MiB) so MB/s figures are comparable.
    group.throughput(Throughput::Bytes(raw_bytes.len() as u64));

    // Full pipeline: simple_packing 24-bit + szip + framing
    group.bench_function("encode_full_pipeline", |b| {
        b.iter(|| {
            encode(&meta, &[(&desc, &raw_bytes)], &opts).expect("encode failed");
        });
    });

    // Pre-encoded path: framing only (no simple_packing, no szip)
    group.bench_function("encode_pre_encoded", |b| {
        b.iter(|| {
            encode_pre_encoded(&meta, &[(&gpu_desc, &gpu_bytes)], &opts)
                .expect("encode_pre_encoded failed");
        });
    });

    group.finish();
}

criterion_group!(benches, bench_encode_paths);
criterion_main!(benches);
