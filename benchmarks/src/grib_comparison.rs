//! GRIB vs Tensogram comparison benchmark.
//!
//! Compares ecCodes CCSDS (GRIB grid_ccsds) packing against Tensogram
//! `simple_packing(24)+szip` on 10M float64 values (24-bit precision).
//!
//! ecCodes is the reference. Requires the ecCodes C library installed and the
//! `eccodes` Cargo feature enabled.
//!
//! # Timing model
//! - **GRIB encode**: time spent in `codes_set_double_array("values", ...)`,
//!   which performs the actual quantisation and compression.
//! - **GRIB decode**: time to create a GRIB handle from raw bytes plus
//!   `codes_get_double_array("values", ...)`, mirroring tensogram's
//!   full decode path.
//! - **Tensogram encode**: `encode_pipeline(bytes, config)`.
//! - **Tensogram decode**: `decode_pipeline(encoded, config)`.

use std::ffi::CString;
use std::os::raw::{c_int, c_long, c_void};
use std::time::Instant;

use tensogram_encodings::pipeline::{decode_pipeline, encode_pipeline};
use tensogram_encodings::simple_packing::compute_params;
use tensogram_encodings::{ByteOrder, CompressionType, EncodingType, FilterType, PipelineConfig};

use crate::datagen::generate_weather_field;
use crate::report::{median_ns, ns_to_ms, BenchmarkResult};
use crate::BenchmarkError;

// AEC_DATA_PREPROCESS constant (value = 1 in libaec).
const AEC_DATA_PREPROCESS: u32 = 1;

// ── ecCodes C API (raw) ───────────────────────────────────────────────────────
//
// We declare just the subset needed for this benchmark. The symbols are
// available because the `eccodes` Rust crate links against libeccodes.

extern "C" {
    fn codes_grib_handle_new_from_samples(ctx: *mut c_void, name: *const i8) -> *mut c_void;
    fn codes_handle_new_from_message_copy(
        ctx: *mut c_void,
        msg: *const c_void,
        size: usize,
    ) -> *mut c_void;
    fn codes_handle_delete(h: *mut c_void) -> c_int;
    fn codes_set_long(h: *mut c_void, key: *const i8, val: c_long) -> c_int;
    fn codes_set_string(
        h: *mut c_void,
        key: *const i8,
        val: *const i8,
        length: *mut usize,
    ) -> c_int;
    fn codes_set_double_array(
        h: *mut c_void,
        key: *const i8,
        vals: *const f64,
        length: usize,
    ) -> c_int;
    fn codes_get_double_array(
        h: *mut c_void,
        key: *const i8,
        vals: *mut f64,
        length: *mut usize,
    ) -> c_int;
    fn codes_get_message(
        h: *mut c_void,
        message: *mut *const c_void,
        message_length: *mut usize,
    ) -> c_int;
}

// ── RAII handle wrapper ───────────────────────────────────────────────────────

/// RAII wrapper around `codes_handle*`.
struct GribHandle(*mut c_void);

impl Drop for GribHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { codes_handle_delete(self.0) };
        }
    }
}

impl GribHandle {
    /// Create a handle from an ecCodes sample template by name.
    fn from_samples(name: &str) -> Result<Self, BenchmarkError> {
        let name_c = CString::new(name).map_err(|e| BenchmarkError(e.to_string()))?;
        let h =
            unsafe { codes_grib_handle_new_from_samples(std::ptr::null_mut(), name_c.as_ptr()) };
        if h.is_null() {
            Err(BenchmarkError(format!(
                "codes_grib_handle_new_from_samples({name}) returned null; \
                 check ECCODES_SAMPLES_PATH or ecCodes installation"
            )))
        } else {
            Ok(GribHandle(h))
        }
    }

    /// Create a handle by copying raw GRIB message bytes.
    fn from_message_copy(msg: &[u8]) -> Result<Self, BenchmarkError> {
        let h = unsafe {
            codes_handle_new_from_message_copy(
                std::ptr::null_mut(),
                msg.as_ptr() as *const c_void,
                msg.len(),
            )
        };
        if h.is_null() {
            Err(BenchmarkError(
                "codes_handle_new_from_message_copy returned null".to_string(),
            ))
        } else {
            Ok(GribHandle(h))
        }
    }

    fn set_long(&self, key: &str, val: i64) -> Result<(), BenchmarkError> {
        let key_c = CString::new(key).map_err(|e| BenchmarkError(e.to_string()))?;
        let rc = unsafe { codes_set_long(self.0, key_c.as_ptr(), val as c_long) };
        if rc != 0 {
            Err(BenchmarkError(format!(
                "codes_set_long({key}, {val}) returned {rc}"
            )))
        } else {
            Ok(())
        }
    }

    fn set_string(&self, key: &str, val: &str) -> Result<(), BenchmarkError> {
        let key_c = CString::new(key).map_err(|e| BenchmarkError(e.to_string()))?;
        let val_c = CString::new(val).map_err(|e| BenchmarkError(e.to_string()))?;
        let mut len = val.len() + 1;
        let rc = unsafe { codes_set_string(self.0, key_c.as_ptr(), val_c.as_ptr(), &mut len) };
        if rc != 0 {
            Err(BenchmarkError(format!(
                "codes_set_string({key}, {val}) returned {rc}"
            )))
        } else {
            Ok(())
        }
    }

    /// Encode (pack) `vals` into this GRIB message.  This is the timed operation.
    fn set_values(&self, vals: &[f64]) -> Result<(), BenchmarkError> {
        let key_c = CString::new("values").map_err(|e| BenchmarkError(e.to_string()))?;
        let rc =
            unsafe { codes_set_double_array(self.0, key_c.as_ptr(), vals.as_ptr(), vals.len()) };
        if rc != 0 {
            Err(BenchmarkError(format!(
                "codes_set_double_array(values, len={}) returned {rc}",
                vals.len()
            )))
        } else {
            Ok(())
        }
    }

    /// Decode (unpack) values from this GRIB message.  This is the timed operation.
    fn get_values(&self, n: usize) -> Result<Vec<f64>, BenchmarkError> {
        let key_c = CString::new("values").map_err(|e| BenchmarkError(e.to_string()))?;
        let mut buf = vec![0.0f64; n];
        let mut len = n;
        let rc =
            unsafe { codes_get_double_array(self.0, key_c.as_ptr(), buf.as_mut_ptr(), &mut len) };
        if rc != 0 {
            Err(BenchmarkError(format!(
                "codes_get_double_array(values) returned {rc}"
            )))
        } else {
            buf.truncate(len);
            Ok(buf)
        }
    }

    /// Copy the raw GRIB message bytes into an owned buffer.
    fn message_bytes(&self) -> Result<Vec<u8>, BenchmarkError> {
        let mut ptr: *const c_void = std::ptr::null();
        let mut size: usize = 0;
        let rc = unsafe { codes_get_message(self.0, &mut ptr, &mut size) };
        if rc != 0 || ptr.is_null() {
            return Err(BenchmarkError(format!("codes_get_message returned {rc}")));
        }
        // The pointer is valid for the lifetime of the handle; copy to owned bytes.
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };
        Ok(bytes.to_vec())
    }
}

// ── Grid utilities ────────────────────────────────────────────────────────────

/// Find (ni, nj) such that ni × nj == n (exact), preferring nearly-square grids.
///
/// Returns (1, n) as a fallback for prime n.
fn factorize(n: usize) -> (usize, usize) {
    let root = (n as f64).sqrt() as usize;
    for i in (1..=root).rev() {
        if n.is_multiple_of(i) {
            let j = n / i;
            return (i, j);
        }
    }
    (1, n)
}

/// Configure a GRIB2 handle for N data points on a regular lat-lon grid.
///
/// Sets packingType and bitsPerValue before returning — caller should then
/// call `set_values()` to pack the actual data.
fn setup_grib_handle(
    n: usize,
    packing_type: &str,
    bits_per_value: i64,
) -> Result<GribHandle, BenchmarkError> {
    let h = GribHandle::from_samples("GRIB2")?;

    let (ni, nj) = factorize(n);

    // Configure a regular lat-lon grid covering ±80° lat, 0-360° lon.
    // Exact bounds matter less than correctness for packing benchmarks.
    let i_inc = 360_000_000i64 / ni as i64; // microdegrees
    let j_inc = 160_000_000i64 / nj as i64;

    h.set_string("gridType", "regular_ll")?;
    h.set_long("Ni", ni as i64)?;
    h.set_long("Nj", nj as i64)?;
    h.set_long("latitudeOfFirstGridPoint", 80_000_000)?;
    h.set_long(
        "latitudeOfLastGridPoint",
        80_000_000 - j_inc * (nj as i64 - 1),
    )?;
    h.set_long("longitudeOfFirstGridPoint", 0)?;
    h.set_long("longitudeOfLastGridPoint", i_inc * (ni as i64 - 1))?;
    h.set_long("iDirectionIncrement", i_inc)?;
    h.set_long("jDirectionIncrement", j_inc)?;

    h.set_long("bitsPerValue", bits_per_value)?;
    // Set packingType last — some packing types require grid to be set first.
    h.set_string("packingType", packing_type)?;

    Ok(h)
}

// ── GRIB timing ───────────────────────────────────────────────────────────────

/// Time GRIB encode+decode for a given packing type.
///
/// Warm-up run discarded; then `iterations` timed runs.
fn time_grib(
    values: &[f64],
    packing_type: &str,
    bits_per_value: i64,
    iterations: usize,
    original_bytes: usize,
) -> Result<BenchmarkResult, BenchmarkError> {
    let n = values.len();

    // ── Warm-up ──────────────────────────────────────────────────────────────
    let warm_h = setup_grib_handle(n, packing_type, bits_per_value)?;
    warm_h.set_values(values)?;
    let warm_bytes = warm_h.message_bytes()?;
    let warm_dec = GribHandle::from_message_copy(&warm_bytes)?;
    warm_dec.get_values(n)?;
    let compressed_bytes = warm_bytes.len();

    // ── Timed iterations ─────────────────────────────────────────────────────
    let mut encode_ns = Vec::with_capacity(iterations);
    let mut decode_ns = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        // Encode: create handle + pack values (set_values is the bottleneck).
        let h = setup_grib_handle(n, packing_type, bits_per_value)?;
        let t0 = Instant::now();
        h.set_values(values)?;
        encode_ns.push(t0.elapsed().as_nanos() as u64);

        let msg = h.message_bytes()?;

        // Decode: parse handle from bytes + unpack values.
        let t0 = Instant::now();
        let dec_h = GribHandle::from_message_copy(&msg)?;
        dec_h.get_values(n)?;
        decode_ns.push(t0.elapsed().as_nanos() as u64);
    }

    Ok(BenchmarkResult {
        name: String::new(), // caller fills this in
        encode_ms: ns_to_ms(median_ns(&mut encode_ns)),
        decode_ms: ns_to_ms(median_ns(&mut decode_ns)),
        compressed_bytes,
        original_bytes,
    })
}

// ── Tensogram timing ──────────────────────────────────────────────────────────

fn time_tensogram_sp_szip(
    values: &[f64],
    bits: u32,
    iterations: usize,
    original_bytes: usize,
) -> Result<BenchmarkResult, BenchmarkError> {
    let num_points = values.len();
    let data_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let params = compute_params(values, bits, 0)
        .map_err(|e| BenchmarkError(format!("compute_params({bits}): {e}")))?;

    let config = PipelineConfig {
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
    };

    let map_err = |e: tensogram_encodings::pipeline::PipelineError| -> BenchmarkError {
        BenchmarkError(e.to_string())
    };

    // Warm-up
    let warm = encode_pipeline(&data_bytes, &config).map_err(map_err)?;
    decode_pipeline(&warm.encoded_bytes, &config).map_err(map_err)?;

    let mut encode_ns = Vec::with_capacity(iterations);
    let mut decode_ns = Vec::with_capacity(iterations);
    let compressed_bytes = warm.encoded_bytes.len();

    for _ in 0..iterations {
        let t0 = Instant::now();
        let result = encode_pipeline(&data_bytes, &config).map_err(map_err)?;
        encode_ns.push(t0.elapsed().as_nanos() as u64);

        let t0 = Instant::now();
        decode_pipeline(&result.encoded_bytes, &config).map_err(map_err)?;
        decode_ns.push(t0.elapsed().as_nanos() as u64);
    }

    Ok(BenchmarkResult {
        name: String::new(), // caller fills this in
        encode_ms: ns_to_ms(median_ns(&mut encode_ns)),
        decode_ms: ns_to_ms(median_ns(&mut decode_ns)),
        compressed_bytes,
        original_bytes,
    })
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Run the GRIB comparison benchmark and return results.
///
/// The first result is `"eccodes grid_ccsds"` (the reference).
/// Called by the binary entry point and by integration tests.
pub fn run_grib_comparison_results(
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

    eprintln!("Generating {num_points} weather-like float64 values (seed={seed})...");
    let values = generate_weather_field(num_points, seed);
    let original_bytes = num_points * 8; // 8 bytes per f64

    let mut results = Vec::new();
    let bits = 24;

    /// Collect a benchmark result, logging progress and handling errors
    /// without aborting the full run.
    fn collect(
        name: &str,
        result: Result<BenchmarkResult, BenchmarkError>,
        original_bytes: usize,
        out: &mut Vec<BenchmarkResult>,
    ) {
        match result {
            Ok(mut r) => {
                r.name = name.to_string();
                eprintln!(" done ({:.1} ms encode)", r.encode_ms);
                out.push(r);
            }
            Err(e) => {
                eprintln!(" FAILED: {e}");
                out.push(BenchmarkResult {
                    name: format!("{name} [ERROR]"),
                    encode_ms: 0.0,
                    decode_ms: 0.0,
                    compressed_bytes: 0,
                    original_bytes,
                });
            }
        }
    }

    eprint!("  eccodes grid_ccsds (24-bit)...");
    collect(
        "eccodes grid_ccsds",
        time_grib(&values, "grid_ccsds", bits, iterations, original_bytes),
        original_bytes,
        &mut results,
    );

    eprint!("  eccodes grid_simple (24-bit)...");
    collect(
        "eccodes grid_simple",
        time_grib(&values, "grid_simple", bits, iterations, original_bytes),
        original_bytes,
        &mut results,
    );

    eprint!("  tensogram sp(24)+szip...");
    collect(
        "tensogram sp(24)+szip",
        time_tensogram_sp_szip(&values, bits as u32, iterations, original_bytes),
        original_bytes,
        &mut results,
    );

    Ok(results)
}
