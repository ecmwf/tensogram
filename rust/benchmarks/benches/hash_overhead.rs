// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Benchmark: hash-while-encoding vs post-hoc hash vs no hash.
//!
//! Measures, for four representative codec combinations, the cost of
//! three strategies:
//!
//! 1. **`no_hash`** — encoding only, `hash_algorithm = None`.  Baseline:
//!    what the pipeline costs without any integrity hash.
//! 2. **`two_pass_hash`** — the pre-optimisation behaviour: encode, then
//!    walk the encoded buffer again with `compute_hash` to build the
//!    descriptor's hash value.  Implemented here explicitly with
//!    `compute_hash = false` + manual post-hoc `xxh3_64`.
//! 3. **`fused_inline_hash`** — the new path: pipeline-level
//!    `compute_hash = true` so the hash is produced inline with the
//!    codec output, eliminating the second read of the encoded buffer.
//!
//! The spread between (2) and (3) is the win the PR claims.  The
//! spread between (1) and (3) is the residual cost of keeping a hash
//! in every message (which we accept as the correctness/integrity
//! contract).
//!
//! Workload: 16 Mi f64 values (~128 MiB raw), smooth sinusoidal data
//! (matches the existing benchmarks for comparability).

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use tensogram_encodings::pipeline::{PipelineConfig, encode_pipeline};
use tensogram_encodings::simple_packing::compute_params;
use tensogram_encodings::{ByteOrder, CompressionType, EncodingType, FilterType};
use xxhash_rust::xxh3::xxh3_64;

const N_FLOATS: usize = 16 * 1024 * 1024; // 16 Mi float64 (~128 MiB raw)

fn synth_data() -> (Vec<f64>, Vec<u8>) {
    let values: Vec<f64> = (0..N_FLOATS)
        .map(|i| (i as f64 * 0.0001).sin() * 1000.0 + 280.0)
        .collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    (values, bytes)
}

fn make_config(
    encoding: EncodingType,
    compression: CompressionType,
    num_values: usize,
    compute_hash: bool,
) -> PipelineConfig {
    PipelineConfig {
        encoding,
        filter: FilterType::None,
        compression,
        num_values,
        byte_order: ByteOrder::Little,
        dtype_byte_width: 8,
        swap_unit_size: 8,
        compression_backend: Default::default(),
        intra_codec_threads: 0,
        compute_hash,
    }
}

fn bench_hash_overhead(c: &mut Criterion) {
    let (values, data_bytes) = synth_data();

    // simple_packing params computed once outside the timed loop — the
    // cost of `compute_params` is not what this benchmark is measuring.
    // `tensogram-benchmarks` depends on `tensogram-encodings` with
    // default features = {szip, zstd, lz4, blosc2, zfp, sz3, threads},
    // so every codec below is always available in this harness.
    let sp_params = compute_params(&values, 24, 0).expect("compute_params");

    let cases: Vec<(&str, EncodingType, CompressionType)> = vec![
        ("none+none", EncodingType::None, CompressionType::None),
        ("none+lz4", EncodingType::None, CompressionType::Lz4),
        (
            "sp24+szip",
            EncodingType::SimplePacking(sp_params.clone()),
            CompressionType::Szip {
                rsi: 128,
                block_size: 16,
                flags: 8, // AEC_DATA_PREPROCESS
                bits_per_sample: 24,
            },
        ),
        (
            "sp24+zstd3",
            EncodingType::SimplePacking(sp_params),
            CompressionType::Zstd { level: 3 },
        ),
    ];

    for (label, encoding, compression) in cases {
        let mut group = c.benchmark_group(format!("hash_overhead/{label}"));
        group.throughput(Throughput::Bytes(data_bytes.len() as u64));

        // (1) Encoding only, no hash at all.
        {
            let config = make_config(encoding.clone(), compression.clone(), N_FLOATS, false);
            group.bench_function("no_hash", |b| {
                b.iter(|| {
                    let r = encode_pipeline(&data_bytes, &config).expect("encode");
                    std::hint::black_box(&r.encoded_bytes);
                });
            });
        }

        // (2) Two-pass: encode without inline hash, then hash the result.
        {
            let config = make_config(encoding.clone(), compression.clone(), N_FLOATS, false);
            group.bench_function("two_pass_hash", |b| {
                b.iter(|| {
                    let r = encode_pipeline(&data_bytes, &config).expect("encode");
                    let h = xxh3_64(&r.encoded_bytes);
                    std::hint::black_box((r.encoded_bytes, h));
                });
            });
        }

        // (3) Fused inline: pipeline computes the hash in lockstep with
        //     the codec output.
        {
            let config = make_config(encoding.clone(), compression.clone(), N_FLOATS, true);
            group.bench_function("fused_inline_hash", |b| {
                b.iter(|| {
                    let r = encode_pipeline(&data_bytes, &config).expect("encode");
                    std::hint::black_box((r.encoded_bytes, r.hash));
                });
            });
        }

        group.finish();
    }
}

criterion_group!(benches, bench_hash_overhead);
criterion_main!(benches);
