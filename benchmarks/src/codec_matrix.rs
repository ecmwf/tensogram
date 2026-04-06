//! Codec matrix benchmark — all encoder × compressor × bit-width combinations.
//!
//! Generates a synthetic weather field, runs every valid pipeline combination,
//! and reports encode time, decode time, compressed size, and compression ratio
//! against the `none+none` baseline.

use std::time::Instant;

// Types re-exported at tensogram_encodings crate root.
use tensogram_encodings::{
    Blosc2Codec, ByteOrder, CompressionType, EncodingType, FilterType, PipelineConfig,
    Sz3ErrorBound, ZfpMode,
};
// Functions and error not re-exported at root — access via the pipeline module.
use tensogram_encodings::pipeline::{decode_pipeline, encode_pipeline};
use tensogram_encodings::simple_packing::compute_params;

use crate::datagen::generate_weather_field;
use crate::report::{median_ns, ns_to_ms, BenchmarkResult};
use crate::BenchmarkError;

// AEC_DATA_PREPROCESS flag value (defined in libaec-sys as 1).
// Using the raw constant avoids adding libaec-sys as a direct dependency.
const AEC_DATA_PREPROCESS: u32 = 1;

// ── Case descriptor ───────────────────────────────────────────────────────────

/// A single benchmark case: a pipeline configuration and a display name.
struct Case {
    name: String,
    config: PipelineConfig,
}

// ── Config builders ───────────────────────────────────────────────────────────

fn none_none(num_points: usize) -> Case {
    Case {
        name: "none+none".to_string(),
        config: PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::None,
            num_values: num_points,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
        },
    }
}

fn none_zstd(num_points: usize) -> Case {
    Case {
        name: "none+zstd(3)".to_string(),
        config: PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::Zstd { level: 3 },
            num_values: num_points,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
        },
    }
}

fn none_lz4(num_points: usize) -> Case {
    Case {
        name: "none+lz4".to_string(),
        config: PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::Lz4,
            num_values: num_points,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
        },
    }
}

fn none_blosc2(num_points: usize) -> Case {
    Case {
        name: "none+blosc2(blosclz)".to_string(),
        config: PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::Blosc2 {
                codec: Blosc2Codec::Blosclz,
                clevel: 5,
                typesize: 8, // f64 = 8 bytes
            },
            num_values: num_points,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
        },
    }
}

fn none_szip(num_points: usize) -> Case {
    // Treat raw f64 bytes as 32-bit samples (two per value).
    // libaec supports bits_per_sample up to 32; raw 64-bit is not supported.
    Case {
        name: "none+szip(32)".to_string(),
        config: PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::Szip {
                rsi: 128,
                block_size: 16,
                flags: AEC_DATA_PREPROCESS,
                bits_per_sample: 32,
            },
            num_values: num_points,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
        },
    }
}

fn sp_none(num_points: usize, bits: u32, values: &[f64]) -> Result<Case, BenchmarkError> {
    let params = compute_params(values, bits, 0)
        .map_err(|e| BenchmarkError(format!("sp({bits}) params: {e}")))?;
    Ok(Case {
        name: format!("sp({bits})+none"),
        config: PipelineConfig {
            encoding: EncodingType::SimplePacking(params),
            filter: FilterType::None,
            compression: CompressionType::None,
            num_values: num_points,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
        },
    })
}

fn sp_zstd(num_points: usize, bits: u32, values: &[f64]) -> Result<Case, BenchmarkError> {
    let params = compute_params(values, bits, 0)
        .map_err(|e| BenchmarkError(format!("sp({bits}) params: {e}")))?;
    Ok(Case {
        name: format!("sp({bits})+zstd(3)"),
        config: PipelineConfig {
            encoding: EncodingType::SimplePacking(params),
            filter: FilterType::None,
            compression: CompressionType::Zstd { level: 3 },
            num_values: num_points,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
        },
    })
}

fn sp_lz4(num_points: usize, bits: u32, values: &[f64]) -> Result<Case, BenchmarkError> {
    let params = compute_params(values, bits, 0)
        .map_err(|e| BenchmarkError(format!("sp({bits}) params: {e}")))?;
    Ok(Case {
        name: format!("sp({bits})+lz4"),
        config: PipelineConfig {
            encoding: EncodingType::SimplePacking(params),
            filter: FilterType::None,
            compression: CompressionType::Lz4,
            num_values: num_points,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
        },
    })
}

fn sp_blosc2(num_points: usize, bits: u32, values: &[f64]) -> Result<Case, BenchmarkError> {
    let params = compute_params(values, bits, 0)
        .map_err(|e| BenchmarkError(format!("sp({bits}) params: {e}")))?;
    // typesize = bytes needed per packed element (div_ceil for non-byte-aligned packing)
    let typesize = (bits as usize).div_ceil(8);
    Ok(Case {
        name: format!("sp({bits})+blosc2(blosclz)"),
        config: PipelineConfig {
            encoding: EncodingType::SimplePacking(params),
            filter: FilterType::None,
            compression: CompressionType::Blosc2 {
                codec: Blosc2Codec::Blosclz,
                clevel: 5,
                typesize,
            },
            num_values: num_points,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
        },
    })
}

fn sp_szip(num_points: usize, bits: u32, values: &[f64]) -> Result<Case, BenchmarkError> {
    let params = compute_params(values, bits, 0)
        .map_err(|e| BenchmarkError(format!("sp({bits}) params: {e}")))?;
    Ok(Case {
        name: format!("sp({bits})+szip"),
        config: PipelineConfig {
            encoding: EncodingType::SimplePacking(params),
            filter: FilterType::None,
            compression: CompressionType::Szip {
                rsi: 128,
                block_size: 16,
                flags: AEC_DATA_PREPROCESS,
                bits_per_sample: bits,
            },
            num_values: num_points,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
        },
    })
}

/// ZFP uses `from_ne_bytes` internally — use little-endian byte order so the
/// bytes match native representation on the benchmark host.
fn none_zfp(num_points: usize, rate: f64) -> Case {
    Case {
        name: format!("none+zfp(rate={rate})"),
        config: PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::Zfp {
                mode: ZfpMode::FixedRate { rate },
            },
            num_values: num_points,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
        },
    }
}

/// SZ3 also interprets bytes as native-endian floats.
fn none_sz3(num_points: usize, tolerance: f64) -> Case {
    Case {
        name: format!("none+sz3(abs={tolerance})"),
        config: PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::Sz3 {
                error_bound: Sz3ErrorBound::Absolute(tolerance),
            },
            num_values: num_points,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
        },
    }
}

// ── Case builder ──────────────────────────────────────────────────────────────

/// Build all benchmark cases in display order.
///
/// The `none+none` case is always first — it serves as the reference.
fn build_cases(num_points: usize, values: &[f64]) -> Result<Vec<Case>, BenchmarkError> {
    let mut cases = Vec::with_capacity(24);

    // Reference
    cases.push(none_none(num_points));

    // Raw f64 + lossless compressors (no encoding)
    cases.push(none_zstd(num_points));
    cases.push(none_lz4(num_points));
    cases.push(none_blosc2(num_points));
    cases.push(none_szip(num_points));

    // simple_packing at 16/24/32 bits + each lossless compressor
    for bits in [16u32, 24, 32] {
        cases.push(sp_none(num_points, bits, values)?);
        cases.push(sp_zstd(num_points, bits, values)?);
        cases.push(sp_lz4(num_points, bits, values)?);
        cases.push(sp_blosc2(num_points, bits, values)?);
        cases.push(sp_szip(num_points, bits, values)?);
    }

    // Lossy floating-point compressors (ZFP fixed-rate, SZ3)
    for rate in [16.0f64, 24.0, 32.0] {
        cases.push(none_zfp(num_points, rate));
    }
    cases.push(none_sz3(num_points, 0.01));

    Ok(cases)
}

// ── Timing ────────────────────────────────────────────────────────────────────

/// Run one benchmark case: warm up, then time `iterations` encode+decode cycles.
///
/// Returns a `BenchmarkResult` with median encode/decode times.
fn run_case(
    case: &Case,
    data_bytes: &[u8],
    original_bytes: usize,
    iterations: usize,
) -> Result<BenchmarkResult, BenchmarkError> {
    let map_err = |e: tensogram_encodings::pipeline::PipelineError| -> BenchmarkError {
        BenchmarkError(e.to_string())
    };

    // Warm-up iteration (result discarded).
    let warm = encode_pipeline(data_bytes, &case.config).map_err(map_err)?;
    drop(decode_pipeline(&warm.encoded_bytes, &case.config).map_err(map_err)?);

    let mut encode_ns = Vec::with_capacity(iterations);
    let mut decode_ns = Vec::with_capacity(iterations);
    let mut compressed_bytes = warm.encoded_bytes.len();

    for _ in 0..iterations {
        let t0 = Instant::now();
        let result = encode_pipeline(data_bytes, &case.config).map_err(map_err)?;
        encode_ns.push(t0.elapsed().as_nanos() as u64);

        compressed_bytes = result.encoded_bytes.len();

        let t0 = Instant::now();
        drop(decode_pipeline(&result.encoded_bytes, &case.config).map_err(map_err)?);
        decode_ns.push(t0.elapsed().as_nanos() as u64);
    }

    Ok(BenchmarkResult {
        name: case.name.clone(),
        encode_ms: ns_to_ms(median_ns(&mut encode_ns)),
        decode_ms: ns_to_ms(median_ns(&mut decode_ns)),
        compressed_bytes,
        original_bytes,
    })
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Run all codec matrix benchmarks and return results.
///
/// Results are ordered with `none+none` first (the reference row).
/// Called by the binary entry point and by integration tests.
pub fn run_codec_matrix_results(
    num_points: usize,
    iterations: usize,
    seed: u64,
) -> Result<Vec<BenchmarkResult>, BenchmarkError> {
    if num_points == 0 {
        return Err(BenchmarkError("num_points must be > 0".to_string()));
    }

    // Round up to the next multiple of 4 so szip cases work at any bit width.
    // libaec promotes 24-bit samples to 4-byte containers, requiring the packed
    // byte count to be a multiple of 4.  Padding by at most 3 values has no
    // measurable impact on benchmark accuracy at production sizes.
    let num_points = num_points.next_multiple_of(4);

    eprintln!("Generating {num_points} weather-like float64 values (seed={seed})...");
    let values = generate_weather_field(num_points, seed);
    // Little-endian bytes (native on x86/ARM64 – matches ZFP/SZ3 expectations).
    let data_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let original_bytes = data_bytes.len();

    eprintln!("Building {num_points} × {} pipeline configs...", 24);
    let cases = build_cases(num_points, &values)?;

    eprintln!(
        "Running {} cases ({iterations} iterations each)...",
        cases.len()
    );
    let mut results = Vec::with_capacity(cases.len());
    for (i, case) in cases.iter().enumerate() {
        eprint!("  [{:2}/{}] {:<35}", i + 1, cases.len(), &case.name);
        match run_case(case, &data_bytes, original_bytes, iterations) {
            Ok(r) => {
                eprintln!(" done ({:.1} ms encode)", r.encode_ms);
                results.push(r);
            }
            Err(e) => {
                // Report failures without aborting the entire benchmark.
                eprintln!(" FAILED: {e}");
                results.push(BenchmarkResult {
                    name: format!("{} [ERROR]", case.name),
                    encode_ms: 0.0,
                    decode_ms: 0.0,
                    compressed_bytes: 0,
                    original_bytes,
                });
            }
        }
    }

    Ok(results)
}
