//! Codec matrix benchmark — all encoder × compressor × bit-width combinations.
//!
//! Generates a synthetic weather field, runs every valid pipeline combination,
//! and reports encode time, decode time, compressed size, and compression ratio
//! against the `none+none` baseline.

use std::time::Instant;

use tensogram_encodings::pipeline::{decode_pipeline, encode_pipeline};
use tensogram_encodings::simple_packing::compute_params;
use tensogram_encodings::{
    Blosc2Codec, ByteOrder, CompressionType, EncodingType, FilterType, PipelineConfig,
    Sz3ErrorBound, ZfpMode,
};

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

// ── Config helpers ────────────────────────────────────────────────────────────

/// Construct a benchmark case with the shared pipeline defaults
/// (no filter, little-endian, f64).
fn make_case(
    name: impl Into<String>,
    encoding: EncodingType,
    compression: CompressionType,
    num_values: usize,
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
        },
    }
}

/// Construct a simple-packing case: compute packing params, then build the case.
fn sp_case(
    bits: u32,
    compressor_name: &str,
    compression: CompressionType,
    num_values: usize,
    values: &[f64],
) -> Result<Case, BenchmarkError> {
    let params = compute_params(values, bits, 0)
        .map_err(|e| BenchmarkError(format!("sp({bits}) params: {e}")))?;
    Ok(make_case(
        format!("sp({bits})+{compressor_name}"),
        EncodingType::SimplePacking(params),
        compression,
        num_values,
    ))
}

// ── Case builder ──────────────────────────────────────────────────────────────

/// Build all benchmark cases in display order.
///
/// The `none+none` case is always first — it serves as the reference.
fn build_cases(n: usize, values: &[f64]) -> Result<Vec<Case>, BenchmarkError> {
    let mut c = Vec::with_capacity(24);

    // ── Baseline ──────────────────────────────────────────────────────────────
    c.push(make_case(
        "none+none",
        EncodingType::None,
        CompressionType::None,
        n,
    ));

    // ── Raw f64 + lossless compressors (no encoding) ──────────────────────────
    c.push(make_case(
        "none+zstd(3)",
        EncodingType::None,
        CompressionType::Zstd { level: 3 },
        n,
    ));
    c.push(make_case(
        "none+lz4",
        EncodingType::None,
        CompressionType::Lz4,
        n,
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
    ));
    // libaec supports up to 32-bit samples; treat raw f64 bytes as two 32-bit samples each.
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
    ));

    // ── simple_packing at 16/24/32 bits + each lossless compressor ────────────
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

    // ── Lossy floating-point compressors ──────────────────────────────────────
    for rate in [16.0f64, 24.0, 32.0] {
        c.push(make_case(
            format!("none+zfp(rate={rate})"),
            EncodingType::None,
            CompressionType::Zfp {
                mode: ZfpMode::FixedRate { rate },
            },
            n,
        ));
    }
    c.push(make_case(
        "none+sz3(abs=0.01)",
        EncodingType::None,
        CompressionType::Sz3 {
            error_bound: Sz3ErrorBound::Absolute(0.01),
        },
        n,
    ));

    Ok(c)
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
///
/// # Note on `num_points` padding
/// The actual number of values encoded is rounded up to the next multiple of 4
/// (by at most 3 values) for szip alignment. The reported sizes and the
/// `original_bytes` field in each result reflect this padded count.
pub fn run_codec_matrix_results(
    num_points: usize,
    iterations: usize,
    seed: u64,
) -> Result<Vec<BenchmarkResult>, BenchmarkError> {
    if num_points == 0 {
        return Err(BenchmarkError("num_points must be > 0".to_string()));
    }
    if iterations == 0 {
        return Err(BenchmarkError("iterations must be > 0".to_string()));
    }

    // Round up to the next multiple of 4 so szip cases work at any bit width.
    // libaec promotes 24-bit samples to 4-byte containers, requiring the packed
    // byte count to be a multiple of 4.  Padding by at most 3 values has no
    // measurable impact on benchmark accuracy at production sizes.
    let num_points = num_points.next_multiple_of(4);

    eprintln!("Generating {num_points} weather-like float64 values (seed={seed})...");
    let values = generate_weather_field(num_points, seed);
    // Little-endian bytes (native on x86/ARM64 — matches ZFP/SZ3 expectations).
    let data_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let original_bytes = data_bytes.len();

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
