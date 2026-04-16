// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Threads-scaling benchmark for the v0.13.0 multi-threaded coding pipeline.
//!
//! Measures encode/decode wall-clock time and throughput for a chosen
//! set of representative codecs across a sweep of `threads` values.
//! Always reports relative to the `threads=0` baseline.

use std::time::Instant;

use tensogram_encodings::pipeline::{decode_pipeline, encode_pipeline, encode_pipeline_f64};
use tensogram_encodings::simple_packing::compute_params;
use tensogram_encodings::{
    Blosc2Codec, ByteOrder, CompressionType, EncodingType, FilterType, PipelineConfig,
};

use crate::constants::AEC_DATA_PREPROCESS;
use crate::datagen::generate_weather_field;
use crate::report::compute_timing_stats;
use crate::BenchmarkError;

type ConfigBuilder = Box<dyn Fn(u32, &[f64]) -> Result<PipelineConfig, BenchmarkError>>;

/// One scaling case: a codec configuration, plus a human-readable name.
struct Case {
    name: String,
    /// Build a fresh `PipelineConfig` for a given thread budget.
    build: ConfigBuilder,
    /// Use the typed f64 fast path?  `true` for simple_packing cases.
    use_f64_path: bool,
}

fn raw_case(name: impl Into<String>, compression: CompressionType, num_values: usize) -> Case {
    let name_str = name.into();
    let compression_owned = compression;
    Case {
        name: name_str.clone(),
        build: Box::new(move |threads, _| {
            Ok(PipelineConfig {
                encoding: EncodingType::None,
                filter: FilterType::None,
                compression: compression_owned.clone(),
                num_values,
                byte_order: ByteOrder::Little,
                dtype_byte_width: 8,
                swap_unit_size: 8,
                compression_backend: Default::default(),
                intra_codec_threads: threads,
            })
        }),
        use_f64_path: false,
    }
}

fn sp_case(
    bits: u32,
    compressor_name: &str,
    compression: CompressionType,
    num_values: usize,
) -> Case {
    let name = format!("sp({bits})+{compressor_name}");
    let compression_owned = compression;
    Case {
        name,
        build: Box::new(move |threads, values: &[f64]| {
            let params = compute_params(values, bits, 0)
                .map_err(|e| BenchmarkError::Pipeline(format!("sp({bits}) params: {e}")))?;
            Ok(PipelineConfig {
                encoding: EncodingType::SimplePacking(params),
                filter: FilterType::None,
                compression: compression_owned.clone(),
                num_values,
                byte_order: ByteOrder::Little,
                dtype_byte_width: 8,
                swap_unit_size: 8,
                compression_backend: Default::default(),
                intra_codec_threads: threads,
            })
        }),
        use_f64_path: true,
    }
}

fn build_cases(n: usize) -> Vec<Case> {
    vec![
        // ── Transparent pipelines (encoded output byte-identical across threads) ──
        raw_case("none+none", CompressionType::None, n),
        raw_case("none+lz4", CompressionType::Lz4, n),
        sp_case(16, "none", CompressionType::None, n),
        sp_case(
            24,
            "szip",
            CompressionType::Szip {
                rsi: 128,
                block_size: 16,
                flags: AEC_DATA_PREPROCESS,
                bits_per_sample: 24,
            },
            n,
        ),
        // ── Opaque pipelines (threads trade off for ratio/speed) ──
        raw_case("none+zstd(3)", CompressionType::Zstd { level: 3 }, n),
        raw_case(
            "none+blosc2(lz4)",
            CompressionType::Blosc2 {
                codec: Blosc2Codec::Lz4,
                clevel: 5,
                typesize: 8,
            },
            n,
        ),
        sp_case(
            24,
            "blosc2(lz4)",
            CompressionType::Blosc2 {
                codec: Blosc2Codec::Lz4,
                clevel: 5,
                typesize: 3,
            },
            n,
        ),
    ]
}

#[derive(Debug, Clone)]
pub struct ThreadsScalingRow {
    pub case: String,
    pub threads: u32,
    pub encode_median_ms: f64,
    pub decode_median_ms: f64,
    pub compressed_bytes: usize,
    pub original_bytes: usize,
}

impl ThreadsScalingRow {
    pub fn ratio_pct(&self) -> f64 {
        if self.original_bytes == 0 {
            0.0
        } else {
            100.0 * self.compressed_bytes as f64 / self.original_bytes as f64
        }
    }

    pub fn encode_mbps(&self) -> f64 {
        if self.encode_median_ms <= 0.0 {
            0.0
        } else {
            (self.original_bytes as f64 / 1_000_000.0) / (self.encode_median_ms / 1000.0)
        }
    }

    pub fn decode_mbps(&self) -> f64 {
        if self.decode_median_ms <= 0.0 {
            0.0
        } else {
            (self.original_bytes as f64 / 1_000_000.0) / (self.decode_median_ms / 1000.0)
        }
    }
}

fn time_case(
    case: &Case,
    values: &[f64],
    raw_bytes: &[u8],
    threads: u32,
    iterations: usize,
    warmup: usize,
) -> Result<ThreadsScalingRow, BenchmarkError> {
    let config = (case.build)(threads, values)?;
    let original_bytes = raw_bytes.len();

    for _ in 0..warmup {
        let out = if case.use_f64_path {
            encode_pipeline_f64(values, &config)?
        } else {
            encode_pipeline(raw_bytes, &config)?
        };
        let _ = decode_pipeline(&out.encoded_bytes, &config, false)?;
    }

    let mut encode_ns = Vec::with_capacity(iterations);
    let mut decode_ns = Vec::with_capacity(iterations);
    let mut last_compressed_len = 0;

    for _ in 0..iterations {
        let t0 = Instant::now();
        let out = if case.use_f64_path {
            encode_pipeline_f64(values, &config)?
        } else {
            encode_pipeline(raw_bytes, &config)?
        };
        encode_ns.push(t0.elapsed().as_nanos() as u64);
        last_compressed_len = out.encoded_bytes.len();

        let t0 = Instant::now();
        let _ = decode_pipeline(&out.encoded_bytes, &config, false)?;
        decode_ns.push(t0.elapsed().as_nanos() as u64);
    }

    let enc_stats = compute_timing_stats(&mut encode_ns);
    let dec_stats = compute_timing_stats(&mut decode_ns);

    Ok(ThreadsScalingRow {
        case: case.name.clone(),
        threads,
        encode_median_ms: enc_stats.median_ms,
        decode_median_ms: dec_stats.median_ms,
        compressed_bytes: last_compressed_len,
        original_bytes,
    })
}

/// Run the threads-scaling benchmark.
///
/// For each case in `cases`, measure encode/decode timing across the
/// given `thread_counts`.  Prints a per-case table to stdout.
pub fn run_threads_scaling(
    num_points: usize,
    iterations: usize,
    warmup: usize,
    seed: u64,
    thread_counts: &[u32],
) -> Result<Vec<ThreadsScalingRow>, BenchmarkError> {
    if num_points == 0 {
        return Err(BenchmarkError::Validation(
            "num_points must be > 0".to_string(),
        ));
    }
    if iterations == 0 {
        return Err(BenchmarkError::Validation(
            "iterations must be > 0".to_string(),
        ));
    }
    if thread_counts.is_empty() {
        return Err(BenchmarkError::Validation(
            "thread_counts must not be empty".to_string(),
        ));
    }

    let num_points = num_points.next_multiple_of(4);

    eprintln!("Generating {num_points} weather-like float64 values (seed={seed})...");
    let values = generate_weather_field(num_points, seed);
    let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let cases = build_cases(num_points);
    eprintln!(
        "Running {} cases \u{d7} {} thread counts ({warmup} warmup + {iterations} timed iterations each)...",
        cases.len(),
        thread_counts.len(),
    );

    let mut rows = Vec::with_capacity(cases.len() * thread_counts.len());

    for case in &cases {
        println!();
        println!(
            "── {} ─────────────────────────────────────────────────────",
            case.name
        );
        println!(
            "  {:>7} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} | {:>8}",
            "threads", "enc (ms)", "dec (ms)", "enc MB/s", "dec MB/s", "ratio", "size (MiB)"
        );
        println!(
            "  {:>7} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} | {:>8}",
            "-------", "--------", "--------", "--------", "--------", "------", "----------"
        );

        // Measure every requested thread count; the first reading is the
        // baseline that subsequent rows are compared against.  Relative
        // speedups are therefore always 1.0× for the first row.
        let mut baseline_enc_ms = f64::NAN;
        let mut baseline_dec_ms = f64::NAN;

        for (row_idx, &t) in thread_counts.iter().enumerate() {
            let row = time_case(case, &values, &raw_bytes, t, iterations, warmup)?;
            if row_idx == 0 {
                baseline_enc_ms = row.encode_median_ms;
                baseline_dec_ms = row.decode_median_ms;
            }
            let enc_speedup = baseline_enc_ms / row.encode_median_ms;
            let dec_speedup = baseline_dec_ms / row.decode_median_ms;
            println!(
                "  {:>7} | {:>10.1} | {:>10.1} | {:>10.0} | {:>10.0} | {:>7.1}% | {:>10.1}  (enc x{:.2}, dec x{:.2})",
                t,
                row.encode_median_ms,
                row.decode_median_ms,
                row.encode_mbps(),
                row.decode_mbps(),
                row.ratio_pct(),
                row.compressed_bytes as f64 / (1024.0 * 1024.0),
                enc_speedup,
                dec_speedup,
            );
            rows.push(row);
        }
    }

    Ok(rows)
}
