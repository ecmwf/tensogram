// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Safe Rust wrapper around ZFP floating-point compression.
//!
//! ZFP compresses floating-point arrays (1D-4D) with configurable
//! rate, precision, or accuracy modes. Fixed-rate mode produces
//! blocks of fixed bit size, enabling O(1) random access.

use crate::compression::CompressionError;
use crate::pipeline::ZfpMode;

fn err(msg: impl Into<String>) -> CompressionError {
    CompressionError::Zfp(msg.into())
}

/// Compress f64 values using ZFP.
///
/// Data is treated as a 1D array of `num_values` doubles.
/// Returns compressed bytes.
pub fn zfp_compress_f64(values: &[f64], mode: &ZfpMode) -> Result<Vec<u8>, CompressionError> {
    let num_values = values.len();
    if num_values == 0 {
        return Ok(Vec::new());
    }

    unsafe {
        let ztype = zfp_sys_cc::zfp_type_zfp_type_double;
        let field =
            zfp_sys_cc::zfp_field_1d(values.as_ptr() as *mut std::ffi::c_void, ztype, num_values);
        if field.is_null() {
            return Err(err("zfp_field_1d failed"));
        }

        let zfp = zfp_sys_cc::zfp_stream_open(std::ptr::null_mut());
        if zfp.is_null() {
            zfp_sys_cc::zfp_field_free(field);
            return Err(err("zfp_stream_open failed"));
        }

        set_mode(zfp, mode, ztype)?;

        let bufsize = zfp_sys_cc::zfp_stream_maximum_size(zfp, field);
        let mut buffer = vec![0u8; bufsize as usize];

        let stream = zfp_sys_cc::stream_open(buffer.as_mut_ptr() as *mut std::ffi::c_void, bufsize);
        zfp_sys_cc::zfp_stream_set_bit_stream(zfp, stream);
        zfp_sys_cc::zfp_stream_rewind(zfp);

        let compressed_size = zfp_sys_cc::zfp_compress(zfp, field);
        if compressed_size == 0 {
            zfp_sys_cc::zfp_field_free(field);
            zfp_sys_cc::zfp_stream_close(zfp);
            zfp_sys_cc::stream_close(stream);
            return Err(err("zfp_compress returned 0"));
        }

        zfp_sys_cc::zfp_field_free(field);
        zfp_sys_cc::zfp_stream_close(zfp);
        zfp_sys_cc::stream_close(stream);

        buffer.truncate(compressed_size as usize);
        Ok(buffer)
    }
}

/// Decompress ZFP-compressed data back to f64 values.
pub fn zfp_decompress_f64(
    compressed: &[u8],
    num_values: usize,
    mode: &ZfpMode,
) -> Result<Vec<f64>, CompressionError> {
    if num_values == 0 {
        return Ok(Vec::new());
    }

    let mut output = vec![0.0f64; num_values];

    unsafe {
        let ztype = zfp_sys_cc::zfp_type_zfp_type_double;
        let field = zfp_sys_cc::zfp_field_1d(
            output.as_mut_ptr() as *mut std::ffi::c_void,
            ztype,
            num_values,
        );
        if field.is_null() {
            return Err(err("zfp_field_1d failed"));
        }

        let zfp = zfp_sys_cc::zfp_stream_open(std::ptr::null_mut());
        if zfp.is_null() {
            zfp_sys_cc::zfp_field_free(field);
            return Err(err("zfp_stream_open failed"));
        }

        set_mode(zfp, mode, ztype)?;

        let stream = zfp_sys_cc::stream_open(
            compressed.as_ptr() as *mut std::ffi::c_void,
            compressed.len(),
        );
        zfp_sys_cc::zfp_stream_set_bit_stream(zfp, stream);
        zfp_sys_cc::zfp_stream_rewind(zfp);

        let ret = zfp_sys_cc::zfp_decompress(zfp, field);
        if ret == 0 {
            zfp_sys_cc::zfp_field_free(field);
            zfp_sys_cc::zfp_stream_close(zfp);
            zfp_sys_cc::stream_close(stream);
            return Err(err("zfp_decompress returned 0"));
        }

        zfp_sys_cc::zfp_field_free(field);
        zfp_sys_cc::zfp_stream_close(zfp);
        zfp_sys_cc::stream_close(stream);
    }

    Ok(output)
}

/// Decompress a range of f64 values from fixed-rate ZFP compressed data.
///
/// In fixed-rate mode, each ZFP block of 4 values compresses to exactly
/// `rate * 4` bits, enabling O(1) random access.
pub fn zfp_decompress_range_f64(
    compressed: &[u8],
    total_values: usize,
    mode: &ZfpMode,
    sample_offset: usize,
    sample_count: usize,
) -> Result<Vec<f64>, CompressionError> {
    // For fixed-rate mode, we could do true block-level seeking.
    // For now, decompress all and slice — ZFP decompression is very fast.
    // True O(1) block access requires ZFP's internal block structure which
    // isn't exposed through the simple compress/decompress API.
    let all = zfp_decompress_f64(compressed, total_values, mode)?;

    let end = sample_offset + sample_count;
    if end > all.len() {
        return Err(err(format!(
            "range ({sample_offset}, {sample_count}) exceeds total values {total_values}"
        )));
    }

    Ok(all[sample_offset..end].to_vec())
}

/// Set the ZFP compression mode on a stream.
unsafe fn set_mode(
    zfp: *mut zfp_sys_cc::zfp_stream,
    mode: &ZfpMode,
    ztype: zfp_sys_cc::zfp_type,
) -> Result<(), CompressionError> {
    match mode {
        ZfpMode::FixedRate { rate } => {
            // dims=1 for 1D, zfp_false=0 for non-aligned
            zfp_sys_cc::zfp_stream_set_rate(zfp, *rate, ztype, 1, 0);
        }
        ZfpMode::FixedPrecision { precision } => {
            zfp_sys_cc::zfp_stream_set_precision(zfp, *precision);
        }
        ZfpMode::FixedAccuracy { tolerance } => {
            let ret = zfp_sys_cc::zfp_stream_set_accuracy(zfp, *tolerance);
            if ret == 0.0 {
                return Err(err("zfp_stream_set_accuracy returned 0"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smooth_data(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (i as f64 / n as f64 * std::f64::consts::PI).sin())
            .collect()
    }

    #[test]
    fn zfp_round_trip_fixed_rate() {
        let values = smooth_data(1024);
        let mode = ZfpMode::FixedRate { rate: 16.0 };
        let compressed = zfp_compress_f64(&values, &mode).unwrap();
        assert!(compressed.len() < values.len() * 8);

        let decompressed = zfp_decompress_f64(&compressed, values.len(), &mode).unwrap();
        assert_eq!(decompressed.len(), values.len());

        // Lossy: check within tolerance
        for (orig, dec) in values.iter().zip(decompressed.iter()) {
            assert!(
                (orig - dec).abs() < 0.1,
                "orig={orig}, dec={dec}, diff={}",
                (orig - dec).abs()
            );
        }
    }

    #[test]
    fn zfp_round_trip_fixed_precision() {
        let values = smooth_data(256);
        let mode = ZfpMode::FixedPrecision { precision: 32 };
        let compressed = zfp_compress_f64(&values, &mode).unwrap();
        let decompressed = zfp_decompress_f64(&compressed, values.len(), &mode).unwrap();

        for (orig, dec) in values.iter().zip(decompressed.iter()) {
            assert!((orig - dec).abs() < 0.001, "orig={orig}, dec={dec}");
        }
    }

    #[test]
    fn zfp_round_trip_fixed_accuracy() {
        let values = smooth_data(256);
        let tol = 1e-6;
        let mode = ZfpMode::FixedAccuracy { tolerance: tol };
        let compressed = zfp_compress_f64(&values, &mode).unwrap();
        let decompressed = zfp_decompress_f64(&compressed, values.len(), &mode).unwrap();

        for (orig, dec) in values.iter().zip(decompressed.iter()) {
            assert!(
                (orig - dec).abs() <= tol,
                "orig={orig}, dec={dec}, diff={}, tol={tol}",
                (orig - dec).abs()
            );
        }
    }

    #[test]
    fn zfp_range_decode() {
        let values = smooth_data(512);
        let mode = ZfpMode::FixedRate { rate: 16.0 };
        let compressed = zfp_compress_f64(&values, &mode).unwrap();

        let full = zfp_decompress_f64(&compressed, values.len(), &mode).unwrap();
        let partial = zfp_decompress_range_f64(&compressed, values.len(), &mode, 100, 200).unwrap();

        assert_eq!(partial.len(), 200);
        assert_eq!(&partial[..], &full[100..300]);
    }

    // ── Coverage: edge cases ─────────────────────────────────────────

    #[test]
    fn zfp_compress_empty() {
        let mode = ZfpMode::FixedRate { rate: 16.0 };
        let result = zfp_compress_f64(&[], &mode).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn zfp_decompress_empty() {
        let mode = ZfpMode::FixedRate { rate: 16.0 };
        let result = zfp_decompress_f64(&[], 0, &mode).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn zfp_range_exceeds_total() {
        let values = smooth_data(128);
        let mode = ZfpMode::FixedRate { rate: 16.0 };
        let compressed = zfp_compress_f64(&values, &mode).unwrap();
        // Request range beyond total values
        let result = zfp_decompress_range_f64(&compressed, values.len(), &mode, 100, 100);
        assert!(result.is_err());
    }

    #[test]
    fn zfp_accuracy_mode_roundtrip() {
        let values = smooth_data(256);
        let mode = ZfpMode::FixedAccuracy { tolerance: 0.01 };
        let compressed = zfp_compress_f64(&values, &mode).unwrap();
        let decoded = zfp_decompress_f64(&compressed, values.len(), &mode).unwrap();
        assert_eq!(decoded.len(), values.len());
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() <= 0.01);
        }
    }

    #[test]
    fn zfp_precision_mode_roundtrip() {
        let values = smooth_data(256);
        let mode = ZfpMode::FixedPrecision { precision: 32 };
        let compressed = zfp_compress_f64(&values, &mode).unwrap();
        let decoded = zfp_decompress_f64(&compressed, values.len(), &mode).unwrap();
        assert_eq!(decoded.len(), values.len());
    }
}
