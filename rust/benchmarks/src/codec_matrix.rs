// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::time::Instant;

use tensogram_encodings::pipeline::{decode_pipeline, encode_pipeline, encode_pipeline_f64};
use tensogram_encodings::simple_packing::compute_params;
use tensogram_encodings::{
    Blosc2Codec, ByteOrder, CompressionType, EncodingType, FilterType, PipelineConfig,
    Sz3ErrorBound, ZfpMode,
};

use crate::constants::AEC_DATA_PREPROCESS;
use crate::datagen::generate_weather_field;
use crate::report::{BenchmarkResult, compute_fidelity, compute_timing_stats};
use crate::{BenchmarkError, BenchmarkRun, CaseFailure};

// ── Case descriptor ──────────────────────────────────────────────────────────

struct Case {
    name: String,
    config: PipelineConfig,
    /// For SimplePacking: bit width for re-deriving params each iteration.
    sp_bits: Option<u32>,
    is_lossy: bool,
}

// ── Config helpers ───────────────────────────────────────────────────────────

fn make_case(
    name: impl Into<String>,
    encoding: EncodingType,
    compression: CompressionType,
    num_values: usize,
    is_lossy: bool,
) -> Case {
    Case {
        name: name.into(),
        config: PipelineConfig {
            encoding,
            filter: FilterType::None,
            compression,
            num_values,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
            swap_unit_size: 8, // f64
            compression_backend: Default::default(),
            intra_codec_threads: 0,
        },
        sp_bits: None,
        is_lossy,
    }
}

fn sp_case(
    bits: u32,
    compressor_name: &str,
    compression: CompressionType,
    num_values: usize,
    values: &[f64],
) -> Result<Case, BenchmarkError> {
    let params = compute_params(values, bits, 0)
        .map_err(|e| BenchmarkError::Pipeline(format!("sp({bits}) params: {e}")))?;
    let mut case = make_case(
        format!("sp({bits})+{compressor_name}"),
        EncodingType::SimplePacking(params),
        compression,
        num_values,
        true,
    );
    case.sp_bits = Some(bits);
    Ok(case)
}

// ── Case builder ─────────────────────────────────────────────────────────────

fn build_cases(n: usize, values: &[f64]) -> Result<Vec<Case>, BenchmarkError> {
    let mut c = Vec::with_capacity(24);

    // ── Baseline ─────────────────────────────────────────────────────────────
    c.push(make_case(
        "none+none",
        EncodingType::None,
        CompressionType::None,
        n,
        false,
    ));

    // ── Raw f64 + lossless compressors ───────────────────────────────────────
    c.push(make_case(
        "none+zstd(3)",
        EncodingType::None,
        CompressionType::Zstd { level: 3 },
        n,
        false,
    ));
    c.push(make_case(
        "none+lz4",
        EncodingType::None,
        CompressionType::Lz4,
        n,
        false,
    ));
    c.push(make_case(
        "none+blosc2(blosclz)",
        EncodingType::None,
        CompressionType::Blosc2 {
            codec: Blosc2Codec::Blosclz,
            clevel: 5,
            typesize: 8,
        },
        n,
        false,
    ));
    c.push(make_case(
        "none+szip(32)",
        EncodingType::None,
        CompressionType::Szip {
            rsi: 128,
            block_size: 16,
            flags: AEC_DATA_PREPROCESS,
            bits_per_sample: 32,
        },
        n,
        false,
    ));

    // ── simple_packing at 16/24/32 bits + each lossless compressor ───────────
    for bits in [16u32, 24, 32] {
        c.push(sp_case(bits, "none", CompressionType::None, n, values)?);
        c.push(sp_case(
            bits,
            "zstd(3)",
            CompressionType::Zstd { level: 3 },
            n,
            values,
        )?);
        c.push(sp_case(bits, "lz4", CompressionType::Lz4, n, values)?);

        let typesize = (bits as usize).div_ceil(8);
        c.push(sp_case(
            bits,
            "blosc2(blosclz)",
            CompressionType::Blosc2 {
                codec: Blosc2Codec::Blosclz,
                clevel: 5,
                typesize,
            },
            n,
            values,
        )?);

        c.push(sp_case(
            bits,
            "szip",
            CompressionType::Szip {
                rsi: 128,
                block_size: 16,
                flags: AEC_DATA_PREPROCESS,
                bits_per_sample: bits,
            },
            n,
            values,
        )?);
    }

    // ── Lossy floating-point compressors ─────────────────────────────────────
    for rate in [16.0f64, 24.0, 32.0] {
        c.push(make_case(
            format!("none+zfp(rate={rate})"),
            EncodingType::None,
            CompressionType::Zfp {
                mode: ZfpMode::FixedRate { rate },
            },
            n,
            true,
        ));
    }
    c.push(make_case(
        "none+sz3(abs=0.01)",
        EncodingType::None,
        CompressionType::Sz3 {
            error_bound: Sz3ErrorBound::Absolute(0.01),
        },
        n,
        true,
    ));

    Ok(c)
}

// ── Timing ───────────────────────────────────────────────────────────────────

fn run_case(
    case: &Case,
    data_bytes: &[u8],
    values: &[f64],
    original_bytes: usize,
    iterations: usize,
    warmup: usize,
) -> Result<BenchmarkResult, BenchmarkError> {
    // For SP cases, compute_params is included in the encode timing.
    let build_config = |recompute_params: bool| -> Result<PipelineConfig, BenchmarkError> {
        if let Some(bits) = case.sp_bits {
            if recompute_params {
                let params = compute_params(values, bits, 0)
                    .map_err(|e| BenchmarkError::Pipeline(format!("sp({bits}) params: {e}")))?;
                Ok(PipelineConfig {
                    encoding: EncodingType::SimplePacking(params),
                    filter: case.config.filter.clone(),
                    compression: case.config.compression.clone(),
                    num_values: case.config.num_values,
                    byte_order: case.config.byte_order,
                    dtype_byte_width: case.config.dtype_byte_width,
                    swap_unit_size: case.config.swap_unit_size,
                    compression_backend: case.config.compression_backend,
                    intra_codec_threads: case.config.intra_codec_threads,
                })
            } else {
                Ok(case.config.clone())
            }
        } else {
            Ok(case.config.clone())
        }
    };

    let use_f64_path = case.sp_bits.is_some();

    for _ in 0..warmup {
        let config = build_config(true)?;
        let encoded = if use_f64_path {
            encode_pipeline_f64(values, &config)?
        } else {
            encode_pipeline(data_bytes, &config)?
        };
        let _ = decode_pipeline(&encoded.encoded_bytes, &config, false)?;
    }

    let mut encode_ns = Vec::with_capacity(iterations);
    let mut decode_ns = Vec::with_capacity(iterations);
    let mut compressed_sizes: Vec<usize> = Vec::with_capacity(iterations);
    let mut last_decoded: Vec<u8> = Vec::new();

    for _ in 0..iterations {
        let t0 = Instant::now();
        let config = build_config(true)?;
        let result = if use_f64_path {
            encode_pipeline_f64(values, &config)?
        } else {
            encode_pipeline(data_bytes, &config)?
        };
        encode_ns.push(t0.elapsed().as_nanos() as u64);

        compressed_sizes.push(result.encoded_bytes.len());

        let t0 = Instant::now();
        let decoded = decode_pipeline(&result.encoded_bytes, &config, false)?;
        decode_ns.push(t0.elapsed().as_nanos() as u64);

        last_decoded = decoded;
    }

    let fidelity = compute_fidelity(data_bytes, &last_decoded, case.is_lossy);
    if matches!(fidelity, crate::report::Fidelity::Unchecked) {
        return Err(BenchmarkError::Pipeline(format!(
            "fidelity could not be computed for '{}' (unexpected decode output length)",
            case.name
        )));
    }
    if !case.is_lossy && !matches!(fidelity, crate::report::Fidelity::Exact) {
        return Err(BenchmarkError::Pipeline(format!(
            "lossless case '{}' did not round-trip exactly: {fidelity:?}",
            case.name
        )));
    }

    let compressed_bytes = compressed_sizes
        .last()
        .copied()
        .expect("iterations validated > 0");
    let compressed_bytes_varied = compressed_sizes.windows(2).any(|w| w[0] != w[1]);

    Ok(BenchmarkResult {
        name: case.name.clone(),
        encode: compute_timing_stats(&mut encode_ns),
        decode: compute_timing_stats(&mut decode_ns),
        compressed_bytes,
        original_bytes,
        compressed_bytes_varied,
        fidelity,
    })
}

// ── Public API ───────────────────────────────────────────────────────────────

pub fn run_codec_matrix(
    num_points: usize,
    iterations: usize,
    warmup: usize,
    seed: u64,
) -> Result<BenchmarkRun, BenchmarkError> {
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
    if warmup == 0 {
        return Err(BenchmarkError::Validation("warmup must be > 0".to_string()));
    }

    // Round up to next multiple of 4 for szip alignment (at most 3 extra values).
    let num_points = num_points.next_multiple_of(4);

    eprintln!("Generating {num_points} weather-like float64 values (seed={seed})...");
    let values = generate_weather_field(num_points, seed);
    let data_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let original_bytes = data_bytes.len();

    let cases = build_cases(num_points, &values)?;
    let total_cases = cases.len();
    eprintln!(
        "Running {total_cases} cases ({warmup} warmup + {iterations} timed iterations each)..."
    );

    let mut results = Vec::with_capacity(total_cases);
    let mut failures = Vec::new();

    for (i, case) in cases.iter().enumerate() {
        eprint!("  [{:2}/{}] {:<35}", i + 1, total_cases, &case.name);
        match run_case(
            case,
            &data_bytes,
            &values,
            original_bytes,
            iterations,
            warmup,
        ) {
            Ok(r) => {
                eprintln!(
                    " {:.1} ms enc, {:.1} ms dec, {:.1}%",
                    r.encode.median_ms,
                    r.decode.median_ms,
                    r.ratio_pct()
                );
                results.push(r);
            }
            Err(e) => {
                eprintln!(" FAILED: {e}");
                failures.push(CaseFailure {
                    name: case.name.clone(),
                    error: e.to_string(),
                });
            }
        }
    }

    Ok(BenchmarkRun {
        results,
        failures,
        total_cases,
    })
}
