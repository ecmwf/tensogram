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
    // Same honest/malformed matrix as szip's `aec_decompress`: only
    // `num_values == 0 && compressed.is_empty()` is a legitimate
    // empty-in / empty-out round-trip; the two mixed cases below are
    // malformed-descriptor symptoms and must not silently return empty.
    match (num_values, compressed.is_empty()) {
        (0, true) => return Ok(Vec::new()),
        (0, false) => {
            return Err(err(
                "num_values=0 with non-empty compressed stream (malformed zfp descriptor)"
                    .to_string(),
            ));
        }
        (_, true) => {
            return Err(err(format!(
                "num_values={num_values} with empty compressed stream (truncated or malformed payload)"
            )));
        }
        _ => {}
    }

    // Fallible reservation: `num_values` flows from the descriptor via
    // `ZfpCompressor`, so an attacker-supplied value must not abort the
    // process through an infallible `vec![0.0f64; N]`. After
    // `try_reserve_exact(num_values)` the capacity is at least
    // `num_values`, so the subsequent `resize` only performs the
    // zero-fill that libzfp then overwrites — no reallocation.
    let mut output: Vec<f64> = Vec::new();
    output.try_reserve_exact(num_values).map_err(|e| {
        err(format!(
            "failed to reserve {} bytes for zfp decompression: {e}",
            num_values.saturating_mul(std::mem::size_of::<f64>()),
        ))
    })?;
    output.resize(num_values, 0.0);

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

        // RAII-free cleanup helper for the early-return paths below.
        let cleanup = |field, zfp, stream: Option<_>| {
            zfp_sys_cc::zfp_field_free(field);
            zfp_sys_cc::zfp_stream_close(zfp);
            if let Some(s) = stream {
                zfp_sys_cc::stream_close(s);
            }
        };

        if let Err(e) = set_mode(zfp, mode, ztype) {
            cleanup(field, zfp, None);
            return Err(e);
        }

        // SECURITY (SEC-009): libzfp's decoder does NOT bounds-check the
        // input bitstream against the requested element count — a
        // truncated stream with a large `num_values` makes
        // `stream_read_word` read past the buffer (ASan SEGV / OOB read
        // / potential info leak), reachable from a hostile `.tgm` with
        // `compression=zfp`.  Defend at the shim: zfp tells us the
        // maximum number of bytes it could read for this field+mode via
        // `zfp_stream_maximum_size`.  We (a) reject a compressed stream
        // larger than that maximum (malformed), and (b) decode from a
        // zero-padded buffer of exactly `max_size` bytes so the decoder
        // can never read past our allocation regardless of how it walks
        // the stream.  The trailing zero padding is just unused
        // bitstream to zfp.
        let max_size = zfp_sys_cc::zfp_stream_maximum_size(zfp, field) as usize;
        if max_size == 0 || compressed.len() > max_size {
            cleanup(field, zfp, None);
            return Err(err(format!(
                "zfp stream length {} inconsistent with descriptor (max {max_size} for \
                 {num_values} values) — truncated or malformed payload",
                compressed.len()
            )));
        }
        // Padded, fallibly-allocated decode buffer.
        let mut padded: Vec<u8> = Vec::new();
        if let Err(e) = padded.try_reserve_exact(max_size) {
            cleanup(field, zfp, None);
            return Err(err(format!(
                "failed to reserve {max_size} bytes for zfp input padding: {e}"
            )));
        }
        padded.resize(max_size, 0);
        padded[..compressed.len()].copy_from_slice(compressed);

        let stream =
            zfp_sys_cc::stream_open(padded.as_mut_ptr() as *mut std::ffi::c_void, max_size);
        zfp_sys_cc::zfp_stream_set_bit_stream(zfp, stream);
        zfp_sys_cc::zfp_stream_rewind(zfp);

        let ret = zfp_sys_cc::zfp_decompress(zfp, field);
        if ret == 0 {
            cleanup(field, zfp, Some(stream));
            return Err(err("zfp_decompress returned 0"));
        }

        cleanup(field, zfp, Some(stream));
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

    let end = sample_offset.checked_add(sample_count).ok_or_else(|| {
        err(format!(
            "range end overflow: sample_offset {sample_offset} + sample_count {sample_count}"
        ))
    })?;
    if end > all.len() {
        return Err(err(format!(
            "range ({sample_offset}, {sample_count}) exceeds total values {total_values}"
        )));
    }

    let mut out: Vec<f64> = Vec::new();
    out.try_reserve_exact(sample_count).map_err(|e| {
        err(format!(
            "failed to reserve {} bytes for zfp range output: {e}",
            sample_count.saturating_mul(std::mem::size_of::<f64>()),
        ))
    })?;
    out.extend_from_slice(&all[sample_offset..end]);
    Ok(out)
}

/// Set the ZFP compression mode on a stream.
unsafe fn set_mode(
    zfp: *mut zfp_sys_cc::zfp_stream,
    mode: &ZfpMode,
    ztype: zfp_sys_cc::zfp_type,
) -> Result<(), CompressionError> {
    unsafe {
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
    fn zfp_decompress_rejects_malformed_size_stream_pairings() {
        let mode = ZfpMode::FixedRate { rate: 16.0 };
        // num_values == 0 but a non-empty stream is a malformed descriptor
        // — must not silently return empty.
        let err = zfp_decompress_f64(&[1u8, 2, 3, 4], 0, &mode)
            .expect_err("num_values=0 with data must be rejected");
        assert!(
            format!("{err}").contains("malformed zfp descriptor"),
            "got: {err}"
        );

        // num_values > 0 but an empty stream is truncated/missing payload.
        let err = zfp_decompress_f64(&[], 128, &mode)
            .expect_err("num_values>0 with empty stream must be rejected");
        assert!(
            format!("{err}").contains("empty compressed stream"),
            "got: {err}"
        );
    }

    #[test]
    fn zfp_range_offset_plus_count_overflows() {
        // sample_offset + sample_count overflowing usize must surface the
        // dedicated checked_add error, not wrap silently.
        let values = smooth_data(128);
        let mode = ZfpMode::FixedRate { rate: 16.0 };
        let compressed = zfp_compress_f64(&values, &mode).unwrap();
        let err = zfp_decompress_range_f64(&compressed, values.len(), &mode, usize::MAX, 1)
            .expect_err("offset+count overflow must be rejected");
        assert!(
            format!("{err}").contains("range end overflow"),
            "expected overflow error, got: {err}"
        );
    }

    #[test]
    fn zfp_range_zero_count_at_end() {
        // A zero-length range at the exact end is valid and yields an
        // empty slice (end == all.len()), covering the extend_from_slice
        // path with an empty range.
        let values = smooth_data(128);
        let mode = ZfpMode::FixedRate { rate: 16.0 };
        let compressed = zfp_compress_f64(&values, &mode).unwrap();
        let out =
            zfp_decompress_range_f64(&compressed, values.len(), &mode, values.len(), 0).unwrap();
        assert!(out.is_empty());
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

    #[test]
    fn zfp_decompress_rejects_pathological_num_values() {
        let mode = ZfpMode::FixedRate { rate: 16.0 };
        let err = zfp_decompress_f64(&[1u8, 2, 3, 4], usize::MAX, &mode)
            .expect_err("usize::MAX num_values must fail the capacity check");
        let msg = format!("{err}");
        assert!(
            msg.contains("failed to reserve"),
            "error should report allocation failure, got: {msg}"
        );
    }

    /// SEC-009 (HIGH, found by `fuzz_codec_decode`): libzfp's decoder
    /// reads the input bitstream without bounds-checking it against the
    /// requested element count.  A truncated stream (e.g. 1 byte) with a
    /// large `num_values` made `stream_read_word` read out of bounds
    /// (ASan SEGV), reachable from a hostile `.tgm` with
    /// `compression=zfp`.  The shim must now reject the inconsistent
    /// stream rather than letting zfp read past the buffer.
    #[test]
    fn sec009_zfp_truncated_stream_rejected_not_oob() {
        let mode = ZfpMode::FixedRate { rate: 16.0 };
        // The exact fuzzer trigger class: 1 byte of "compressed" data
        // but a descriptor claiming 131072 values.  The security
        // invariant is "no out-of-bounds read / SEGV / abort"; the shim
        // decodes from a zero-padded buffer sized to zfp's maximum, so
        // the call returns safely (Ok with garbage, or a structured
        // Err) and never reads past the input.
        let _ = zfp_decompress_f64(&[87u8], 131072, &mode);

        // A stream LARGER than zfp's maximum for the claimed count is
        // categorically malformed and is rejected.
        let err = zfp_decompress_f64(&vec![0u8; 1_000_000], 4, &mode)
            .expect_err("an over-long stream must be rejected as inconsistent");
        assert!(
            format!("{err}").contains("inconsistent"),
            "expected a stream-length inconsistency error, got: {err}"
        );

        // Sweep a range of (tiny stream, large count) combinations across
        // all modes — none may read out of bounds.
        for mode in [
            ZfpMode::FixedRate { rate: 16.0 },
            ZfpMode::FixedPrecision { precision: 32 },
            ZfpMode::FixedAccuracy { tolerance: 1e-6 },
        ] {
            for &stream_len in &[1usize, 2, 7, 8, 16] {
                for &count in &[1024usize, 65536, 1 << 20] {
                    let tiny = vec![0xABu8; stream_len];
                    // Must return (Ok or Err) without OOB / abort.
                    let _ = zfp_decompress_f64(&tiny, count, &mode);
                }
            }
        }
    }
}
