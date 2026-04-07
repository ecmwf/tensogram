use std::ffi::CString;
use std::os::raw::{c_int, c_long, c_void};
use std::time::Instant;

use tensogram_encodings::pipeline::{decode_pipeline, encode_pipeline_f64};
use tensogram_encodings::simple_packing::compute_params;
use tensogram_encodings::{ByteOrder, CompressionType, EncodingType, FilterType, PipelineConfig};

use crate::constants::AEC_DATA_PREPROCESS;
use crate::datagen::generate_weather_field;
use crate::report::{compute_fidelity, compute_timing_stats, BenchmarkResult};
use crate::{BenchmarkError, BenchmarkRun, CaseFailure};

// ── ecCodes C API (raw) ──────────────────────────────────────────────────────

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

// ── RAII handle wrapper ──────────────────────────────────────────────────────

struct GribHandle(*mut c_void);

impl Drop for GribHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { codes_handle_delete(self.0) };
        }
    }
}

impl GribHandle {
    fn from_samples(name: &str) -> Result<Self, BenchmarkError> {
        let name_c = CString::new(name)
            .map_err(|e| BenchmarkError::Pipeline(format!("invalid sample name '{name}': {e}")))?;
        let h =
            unsafe { codes_grib_handle_new_from_samples(std::ptr::null_mut(), name_c.as_ptr()) };
        if h.is_null() {
            Err(BenchmarkError::Pipeline(format!(
                "codes_grib_handle_new_from_samples({name}) returned null; \
                 check ECCODES_SAMPLES_PATH or ecCodes installation"
            )))
        } else {
            Ok(GribHandle(h))
        }
    }

    fn from_message_copy(msg: &[u8]) -> Result<Self, BenchmarkError> {
        let h = unsafe {
            codes_handle_new_from_message_copy(
                std::ptr::null_mut(),
                msg.as_ptr() as *const c_void,
                msg.len(),
            )
        };
        if h.is_null() {
            Err(BenchmarkError::Pipeline(
                "codes_handle_new_from_message_copy returned null".to_string(),
            ))
        } else {
            Ok(GribHandle(h))
        }
    }

    fn set_long(&self, key: &str, val: i64) -> Result<(), BenchmarkError> {
        let key_c = CString::new(key)
            .map_err(|e| BenchmarkError::Pipeline(format!("invalid key '{key}': {e}")))?;
        let rc = unsafe { codes_set_long(self.0, key_c.as_ptr(), val as c_long) };
        if rc != 0 {
            Err(BenchmarkError::Pipeline(format!(
                "codes_set_long({key}, {val}) returned {rc}"
            )))
        } else {
            Ok(())
        }
    }

    fn set_string(&self, key: &str, val: &str) -> Result<(), BenchmarkError> {
        let key_c = CString::new(key)
            .map_err(|e| BenchmarkError::Pipeline(format!("invalid key '{key}': {e}")))?;
        let val_c = CString::new(val).map_err(|e| {
            BenchmarkError::Pipeline(format!("invalid value '{val}' for key '{key}': {e}"))
        })?;
        let mut len = val.len() + 1;
        let rc = unsafe { codes_set_string(self.0, key_c.as_ptr(), val_c.as_ptr(), &mut len) };
        if rc != 0 {
            Err(BenchmarkError::Pipeline(format!(
                "codes_set_string({key}, {val}) returned {rc}"
            )))
        } else {
            Ok(())
        }
    }

    fn set_values(&self, vals: &[f64]) -> Result<(), BenchmarkError> {
        let key_c = CString::new("values")
            .map_err(|e| BenchmarkError::Pipeline(format!("invalid key 'values': {e}")))?;
        let rc =
            unsafe { codes_set_double_array(self.0, key_c.as_ptr(), vals.as_ptr(), vals.len()) };
        if rc != 0 {
            Err(BenchmarkError::Pipeline(format!(
                "codes_set_double_array(values, len={}) returned {rc}",
                vals.len()
            )))
        } else {
            Ok(())
        }
    }

    fn get_values(&self, n: usize) -> Result<Vec<f64>, BenchmarkError> {
        let key_c = CString::new("values")
            .map_err(|e| BenchmarkError::Pipeline(format!("invalid key 'values': {e}")))?;
        let mut buf = vec![0.0f64; n];
        let mut len = n;
        let rc =
            unsafe { codes_get_double_array(self.0, key_c.as_ptr(), buf.as_mut_ptr(), &mut len) };
        if rc != 0 {
            Err(BenchmarkError::Pipeline(format!(
                "codes_get_double_array(values) returned {rc}"
            )))
        } else {
            buf.truncate(len);
            Ok(buf)
        }
    }

    fn message_bytes(&self) -> Result<Vec<u8>, BenchmarkError> {
        let mut ptr: *const c_void = std::ptr::null();
        let mut size: usize = 0;
        let rc = unsafe { codes_get_message(self.0, &mut ptr, &mut size) };
        if rc != 0 || ptr.is_null() {
            return Err(BenchmarkError::Pipeline(format!(
                "codes_get_message returned {rc}"
            )));
        }
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };
        Ok(bytes.to_vec())
    }
}

// ── Grid utilities ───────────────────────────────────────────────────────────

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

fn setup_grib_handle(
    n: usize,
    packing_type: &str,
    bits_per_value: i64,
) -> Result<GribHandle, BenchmarkError> {
    let h = GribHandle::from_samples("GRIB2")?;

    let (ni, nj) = factorize(n);

    let i_inc = 360_000_000i64 / ni as i64;
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
    h.set_string("packingType", packing_type)?;

    Ok(h)
}

// ── GRIB timing (end-to-end: setup + pack/unpack + serialize) ────────────────

fn time_grib(
    values: &[f64],
    packing_type: &str,
    bits_per_value: i64,
    iterations: usize,
    warmup: usize,
    original_bytes: usize,
) -> Result<BenchmarkResult, BenchmarkError> {
    let n = values.len();

    for _ in 0..warmup {
        let h = setup_grib_handle(n, packing_type, bits_per_value)?;
        h.set_values(values)?;
        let msg = h.message_bytes()?;
        let dec_h = GribHandle::from_message_copy(&msg)?;
        let _ = dec_h.get_values(n)?;
    }

    // End-to-end encode: handle setup + set_values + message_bytes.
    // End-to-end decode: from_message_copy + get_values.
    let mut encode_ns = Vec::with_capacity(iterations);
    let mut decode_ns = Vec::with_capacity(iterations);
    let mut compressed_sizes: Vec<usize> = Vec::with_capacity(iterations);
    let mut last_decoded_vals: Vec<f64> = Vec::new();

    for _ in 0..iterations {
        let t0 = Instant::now();
        let h = setup_grib_handle(n, packing_type, bits_per_value)?;
        h.set_values(values)?;
        let msg = h.message_bytes()?;
        encode_ns.push(t0.elapsed().as_nanos() as u64);

        compressed_sizes.push(msg.len());

        let t0 = Instant::now();
        let dec_h = GribHandle::from_message_copy(&msg)?;
        last_decoded_vals = dec_h.get_values(n)?;
        decode_ns.push(t0.elapsed().as_nanos() as u64);
    }

    let orig_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let dec_bytes: Vec<u8> = last_decoded_vals
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let fidelity = compute_fidelity(&orig_bytes, &dec_bytes, true);
    if matches!(fidelity, crate::report::Fidelity::Unchecked) {
        return Err(BenchmarkError::Pipeline(
            "fidelity could not be computed (unexpected decode output length)".to_string(),
        ));
    }

    let compressed_bytes = compressed_sizes
        .last()
        .copied()
        .expect("iterations validated > 0");
    let compressed_bytes_varied = compressed_sizes.windows(2).any(|w| w[0] != w[1]);

    Ok(BenchmarkResult {
        name: String::new(),
        encode: compute_timing_stats(&mut encode_ns),
        decode: compute_timing_stats(&mut decode_ns),
        compressed_bytes,
        original_bytes,
        compressed_bytes_varied,
        fidelity,
    })
}

// ── Tensogram timing (end-to-end: compute_params + encode/decode pipeline) ───

fn time_tensogram_sp_szip(
    values: &[f64],
    bits: u32,
    iterations: usize,
    warmup: usize,
    original_bytes: usize,
) -> Result<BenchmarkResult, BenchmarkError> {
    let data_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let num_points = values.len();

    let build_config = |vals: &[f64]| -> Result<PipelineConfig, BenchmarkError> {
        let params = compute_params(vals, bits, 0)
            .map_err(|e| BenchmarkError::Pipeline(format!("compute_params({bits}): {e}")))?;
        Ok(PipelineConfig {
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
        })
    };

    for _ in 0..warmup {
        let config = build_config(values)?;
        let encoded = encode_pipeline_f64(values, &config)?;
        let _ = decode_pipeline(&encoded.encoded_bytes, &config)?;
    }

    let mut encode_ns = Vec::with_capacity(iterations);
    let mut decode_ns = Vec::with_capacity(iterations);
    let mut compressed_sizes: Vec<usize> = Vec::with_capacity(iterations);
    let mut last_decoded: Vec<u8> = Vec::new();

    for _ in 0..iterations {
        let t0 = Instant::now();
        let config = build_config(values)?;
        let result = encode_pipeline_f64(values, &config)?;
        encode_ns.push(t0.elapsed().as_nanos() as u64);

        compressed_sizes.push(result.encoded_bytes.len());

        let t0 = Instant::now();
        let decoded = decode_pipeline(&result.encoded_bytes, &config)?;
        decode_ns.push(t0.elapsed().as_nanos() as u64);

        last_decoded = decoded;
    }

    let fidelity = compute_fidelity(&data_bytes, &last_decoded, true);
    if matches!(fidelity, crate::report::Fidelity::Unchecked) {
        return Err(BenchmarkError::Pipeline(
            "fidelity could not be computed (unexpected decode output length)".to_string(),
        ));
    }

    let compressed_bytes = compressed_sizes
        .last()
        .copied()
        .expect("iterations validated > 0");
    let compressed_bytes_varied = compressed_sizes.windows(2).any(|w| w[0] != w[1]);

    Ok(BenchmarkResult {
        name: String::new(),
        encode: compute_timing_stats(&mut encode_ns),
        decode: compute_timing_stats(&mut decode_ns),
        compressed_bytes,
        original_bytes,
        compressed_bytes_varied,
        fidelity,
    })
}

// ── Public API ───────────────────────────────────────────────────────────────

pub fn run_grib_comparison(
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

    eprintln!("Generating {num_points} weather-like float64 values (seed={seed})...");
    let values = generate_weather_field(num_points, seed);
    let original_bytes = num_points * 8;

    let bits = 24i64;
    let total_cases = 3;
    let mut results = Vec::with_capacity(total_cases);
    let mut failures = Vec::new();

    fn collect(
        name: &str,
        result: Result<BenchmarkResult, BenchmarkError>,
        results: &mut Vec<BenchmarkResult>,
        failures: &mut Vec<CaseFailure>,
    ) {
        match result {
            Ok(mut r) => {
                r.name = name.to_string();
                eprintln!(
                    " {:.1} ms enc, {:.1} ms dec",
                    r.encode.median_ms, r.decode.median_ms
                );
                results.push(r);
            }
            Err(e) => {
                eprintln!(" FAILED: {e}");
                failures.push(CaseFailure {
                    name: name.to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    eprint!("  eccodes grid_ccsds (24-bit)...");
    collect(
        "eccodes grid_ccsds",
        time_grib(
            &values,
            "grid_ccsds",
            bits,
            iterations,
            warmup,
            original_bytes,
        ),
        &mut results,
        &mut failures,
    );

    eprint!("  eccodes grid_simple (24-bit)...");
    collect(
        "eccodes grid_simple",
        time_grib(
            &values,
            "grid_simple",
            bits,
            iterations,
            warmup,
            original_bytes,
        ),
        &mut results,
        &mut failures,
    );

    eprint!("  tensogram sp(24)+szip...");
    collect(
        "tensogram sp(24)+szip",
        time_tensogram_sp_szip(&values, bits as u32, iterations, warmup, original_bytes),
        &mut results,
        &mut failures,
    );

    Ok(BenchmarkRun {
        results,
        failures,
        total_cases,
    })
}
