// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Pipeline-independent strict-finite check for raw float input.
//!
//! When [`EncodeOptions::reject_nan`](crate::EncodeOptions::reject_nan)
//! or [`EncodeOptions::reject_inf`](crate::EncodeOptions::reject_inf) is
//! set, this scan runs **before** any encoding/filter/compression stage
//! and bails out on the first NaN or Inf it sees.
//!
//! The guarantee is pipeline-independent — callers get the same
//! contract regardless of which pipeline they picked.  See
//! `plans/RESEARCH_NAN_HANDLING.md` §4.1 for motivation and the
//! user-visible corners this closes.
//!
//! ## Parallelism
//!
//! When `parallel == true` and the `threads` feature is enabled, the
//! scan is split into 64 KiB chunks processed via `rayon::par_chunks`.
//! The reported element index is the first NaN/Inf that the worker
//! handling its chunk saw — **not necessarily the globally first**.
//! This matches the determinism contract used by
//! `tensogram_encodings::simple_packing::compute_params_with_threads`.
//! Sequential callers (`parallel == false`) always get the globally
//! first index.
//!
//! ## Non-float dtypes
//!
//! Integer and bitmask dtypes have no NaN/Inf representation by
//! construction, so the scan short-circuits to `Ok(())` for them —
//! zero cost regardless of flags.

use crate::dtype::Dtype;
use crate::error::{Result, TensogramError};
use tensogram_encodings::ByteOrder;

/// Parallel chunk size in bytes.  Chosen to:
/// - be a multiple of every scalar size in use (16, 8, 4, 2, 1) so the
///   chunk boundary never splits an element;
/// - be small enough that a single worker finishes a chunk quickly,
///   giving early exit a tight latency bound.
const PAR_CHUNK_BYTES: usize = 64 * 1024;

/// Scan `data`, interpreted as elements of `dtype` laid out in
/// `byte_order`, for NaN (if `reject_nan`) and Inf (if `reject_inf`).
///
/// Returns [`TensogramError::Encoding`] on first occurrence with a
/// message that names the kind (`"strict-NaN"` / `"strict-Inf"`), the
/// element index, the dtype, and — for complex types — the component
/// (`"real"` / `"imaginary"`) that triggered the rejection.
///
/// Short-circuits to `Ok(())` when both flags are false, and when the
/// dtype is integer or bitmask.
pub(crate) fn scan(
    data: &[u8],
    dtype: Dtype,
    byte_order: ByteOrder,
    reject_nan: bool,
    reject_inf: bool,
    parallel: bool,
) -> Result<()> {
    if !reject_nan && !reject_inf {
        return Ok(());
    }
    // Use the parallel path only when the feature is compiled in and
    // the caller asked for it AND the input is large enough.  Small
    // payloads are always scanned sequentially.
    let go_parallel = parallel && data.len() >= PAR_CHUNK_BYTES;

    // Single exhaustive match: float dtypes dispatch to their scanner,
    // integer and bitmask short-circuit.  Keeping the integer / bitmask
    // arms explicit (rather than `_ =>`) makes adding a future dtype a
    // compile-time decision point — the author must consciously choose
    // whether it's scannable.
    match dtype {
        Dtype::Float32 => scan_dispatch(
            data,
            4,
            byte_order,
            reject_nan,
            reject_inf,
            go_parallel,
            scan_f32_seq,
        ),
        Dtype::Float64 => scan_dispatch(
            data,
            8,
            byte_order,
            reject_nan,
            reject_inf,
            go_parallel,
            scan_f64_seq,
        ),
        Dtype::Float16 => scan_dispatch(
            data,
            2,
            byte_order,
            reject_nan,
            reject_inf,
            go_parallel,
            scan_f16_seq,
        ),
        Dtype::Bfloat16 => scan_dispatch(
            data,
            2,
            byte_order,
            reject_nan,
            reject_inf,
            go_parallel,
            scan_bf16_seq,
        ),
        Dtype::Complex64 => scan_dispatch(
            data,
            8,
            byte_order,
            reject_nan,
            reject_inf,
            go_parallel,
            scan_complex64_seq,
        ),
        Dtype::Complex128 => scan_dispatch(
            data,
            16,
            byte_order,
            reject_nan,
            reject_inf,
            go_parallel,
            scan_complex128_seq,
        ),
        Dtype::Int8
        | Dtype::Int16
        | Dtype::Int32
        | Dtype::Int64
        | Dtype::Uint8
        | Dtype::Uint16
        | Dtype::Uint32
        | Dtype::Uint64
        | Dtype::Bitmask => Ok(()),
    }
}

/// Dispatches a per-dtype sequential scan across sequential or parallel
/// chunks depending on `go_parallel`.  `element_size` is the byte width
/// of one element (or, for complex, the pair stride).
#[inline]
fn scan_dispatch<F>(
    data: &[u8],
    element_size: usize,
    byte_order: ByteOrder,
    reject_nan: bool,
    reject_inf: bool,
    go_parallel: bool,
    seq_scan: F,
) -> Result<()>
where
    F: Fn(&[u8], ByteOrder, bool, bool, usize) -> Result<()> + Send + Sync + Copy,
{
    if !go_parallel {
        return seq_scan(data, byte_order, reject_nan, reject_inf, 0);
    }

    #[cfg(feature = "threads")]
    {
        use rayon::prelude::*;
        // PAR_CHUNK_BYTES is a multiple of 16, 8, 4, 2, 1 so chunk
        // boundaries are element-aligned for every supported dtype.
        let chunk_elements = PAR_CHUNK_BYTES / element_size;
        data.par_chunks(PAR_CHUNK_BYTES)
            .enumerate()
            .try_for_each(|(chunk_idx, chunk)| {
                let base = chunk_idx * chunk_elements;
                seq_scan(chunk, byte_order, reject_nan, reject_inf, base)
            })
    }
    #[cfg(not(feature = "threads"))]
    {
        // Feature off: fall back to sequential.  Silent fallback is
        // the established pattern elsewhere in the parallel module.
        let _ = element_size; // keep the signature unified
        seq_scan(data, byte_order, reject_nan, reject_inf, 0)
    }
}

// ── Sequential scanners, one per dtype ─────────────────────────────────────

fn scan_f32_seq(
    data: &[u8],
    byte_order: ByteOrder,
    reject_nan: bool,
    reject_inf: bool,
    base_index: usize,
) -> Result<()> {
    // Slice-pattern destructuring below is infallible under
    // `chunks_exact`'s spec; the `else` branches in every scanner are
    // defensive no-op fallbacks, not expected paths.
    for (i, chunk) in data.chunks_exact(4).enumerate() {
        let &[b0, b1, b2, b3] = chunk else {
            continue;
        };
        let bytes = [b0, b1, b2, b3];
        let val = match byte_order {
            ByteOrder::Big => f32::from_be_bytes(bytes),
            ByteOrder::Little => f32::from_le_bytes(bytes),
        };
        if reject_nan && val.is_nan() {
            return Err(nan_err(base_index + i, "float32", None));
        }
        if reject_inf && val.is_infinite() {
            return Err(inf_err(
                base_index + i,
                "float32",
                None,
                val.is_sign_positive(),
            ));
        }
    }
    Ok(())
}

fn scan_f64_seq(
    data: &[u8],
    byte_order: ByteOrder,
    reject_nan: bool,
    reject_inf: bool,
    base_index: usize,
) -> Result<()> {
    for (i, chunk) in data.chunks_exact(8).enumerate() {
        let &[b0, b1, b2, b3, b4, b5, b6, b7] = chunk else {
            continue;
        };
        let bytes = [b0, b1, b2, b3, b4, b5, b6, b7];
        let val = match byte_order {
            ByteOrder::Big => f64::from_be_bytes(bytes),
            ByteOrder::Little => f64::from_le_bytes(bytes),
        };
        if reject_nan && val.is_nan() {
            return Err(nan_err(base_index + i, "float64", None));
        }
        if reject_inf && val.is_infinite() {
            return Err(inf_err(
                base_index + i,
                "float64",
                None,
                val.is_sign_positive(),
            ));
        }
    }
    Ok(())
}

fn scan_f16_seq(
    data: &[u8],
    byte_order: ByteOrder,
    reject_nan: bool,
    reject_inf: bool,
    base_index: usize,
) -> Result<()> {
    // IEEE 754 half: sign(1) + exponent(5) + mantissa(10).
    // exp == 0x1F → Inf (mantissa 0) or NaN (mantissa != 0).
    for (i, chunk) in data.chunks_exact(2).enumerate() {
        let &[b0, b1] = chunk else { continue };
        let bytes = [b0, b1];
        let bits = match byte_order {
            ByteOrder::Big => u16::from_be_bytes(bytes),
            ByteOrder::Little => u16::from_le_bytes(bytes),
        };
        let exp = (bits >> 10) & 0x1F;
        if exp != 0x1F {
            continue;
        }
        let mantissa = bits & 0x03FF;
        if mantissa != 0 {
            if reject_nan {
                return Err(nan_err(base_index + i, "float16", None));
            }
        } else if reject_inf {
            let positive = (bits & 0x8000) == 0;
            return Err(inf_err(base_index + i, "float16", None, positive));
        }
    }
    Ok(())
}

fn scan_bf16_seq(
    data: &[u8],
    byte_order: ByteOrder,
    reject_nan: bool,
    reject_inf: bool,
    base_index: usize,
) -> Result<()> {
    // BFloat16: sign(1) + exponent(8) + mantissa(7).
    // exp == 0xFF → Inf (mantissa 0) or NaN (mantissa != 0).
    for (i, chunk) in data.chunks_exact(2).enumerate() {
        let &[b0, b1] = chunk else { continue };
        let bytes = [b0, b1];
        let bits = match byte_order {
            ByteOrder::Big => u16::from_be_bytes(bytes),
            ByteOrder::Little => u16::from_le_bytes(bytes),
        };
        let exp = (bits >> 7) & 0xFF;
        if exp != 0xFF {
            continue;
        }
        let mantissa = bits & 0x7F;
        if mantissa != 0 {
            if reject_nan {
                return Err(nan_err(base_index + i, "bfloat16", None));
            }
        } else if reject_inf {
            let positive = (bits & 0x8000) == 0;
            return Err(inf_err(base_index + i, "bfloat16", None, positive));
        }
    }
    Ok(())
}

fn scan_complex64_seq(
    data: &[u8],
    byte_order: ByteOrder,
    reject_nan: bool,
    reject_inf: bool,
    base_index: usize,
) -> Result<()> {
    // complex64 = (f32 real, f32 imaginary) = 8 bytes.
    for (i, chunk) in data.chunks_exact(8).enumerate() {
        let &[r0, r1, r2, r3, i0, i1, i2, i3] = chunk else {
            continue;
        };
        let real_bytes = [r0, r1, r2, r3];
        let imag_bytes = [i0, i1, i2, i3];
        let (real, imag) = match byte_order {
            ByteOrder::Big => (
                f32::from_be_bytes(real_bytes),
                f32::from_be_bytes(imag_bytes),
            ),
            ByteOrder::Little => (
                f32::from_le_bytes(real_bytes),
                f32::from_le_bytes(imag_bytes),
            ),
        };
        if reject_nan {
            if real.is_nan() {
                return Err(nan_err(base_index + i, "complex64", Some("real")));
            }
            if imag.is_nan() {
                return Err(nan_err(base_index + i, "complex64", Some("imaginary")));
            }
        }
        if reject_inf {
            if real.is_infinite() {
                return Err(inf_err(
                    base_index + i,
                    "complex64",
                    Some("real"),
                    real.is_sign_positive(),
                ));
            }
            if imag.is_infinite() {
                return Err(inf_err(
                    base_index + i,
                    "complex64",
                    Some("imaginary"),
                    imag.is_sign_positive(),
                ));
            }
        }
    }
    Ok(())
}

fn scan_complex128_seq(
    data: &[u8],
    byte_order: ByteOrder,
    reject_nan: bool,
    reject_inf: bool,
    base_index: usize,
) -> Result<()> {
    // complex128 = (f64 real, f64 imaginary) = 16 bytes.
    for (i, chunk) in data.chunks_exact(16).enumerate() {
        let &[
            r0,
            r1,
            r2,
            r3,
            r4,
            r5,
            r6,
            r7,
            i0,
            i1,
            i2,
            i3,
            i4,
            i5,
            i6,
            i7,
        ] = chunk
        else {
            continue;
        };
        let real_bytes = [r0, r1, r2, r3, r4, r5, r6, r7];
        let imag_bytes = [i0, i1, i2, i3, i4, i5, i6, i7];
        let (real, imag) = match byte_order {
            ByteOrder::Big => (
                f64::from_be_bytes(real_bytes),
                f64::from_be_bytes(imag_bytes),
            ),
            ByteOrder::Little => (
                f64::from_le_bytes(real_bytes),
                f64::from_le_bytes(imag_bytes),
            ),
        };
        if reject_nan {
            if real.is_nan() {
                return Err(nan_err(base_index + i, "complex128", Some("real")));
            }
            if imag.is_nan() {
                return Err(nan_err(base_index + i, "complex128", Some("imaginary")));
            }
        }
        if reject_inf {
            if real.is_infinite() {
                return Err(inf_err(
                    base_index + i,
                    "complex128",
                    Some("real"),
                    real.is_sign_positive(),
                ));
            }
            if imag.is_infinite() {
                return Err(inf_err(
                    base_index + i,
                    "complex128",
                    Some("imaginary"),
                    imag.is_sign_positive(),
                ));
            }
        }
    }
    Ok(())
}

// ── Error message builders ─────────────────────────────────────────────────

fn nan_err(index: usize, dtype: &str, component: Option<&str>) -> TensogramError {
    let suffix = component
        .map(|c| format!(" ({c} component)"))
        .unwrap_or_default();
    TensogramError::Encoding(format!(
        "strict-NaN check: NaN at element {index} of {dtype} array{suffix}"
    ))
}

fn inf_err(index: usize, dtype: &str, component: Option<&str>, positive: bool) -> TensogramError {
    let sign = if positive { "+Inf" } else { "-Inf" };
    let suffix = component
        .map(|c| format!(" ({c} component)"))
        .unwrap_or_default();
    TensogramError::Encoding(format!(
        "strict-Inf check: {sign} at element {index} of {dtype} array{suffix}"
    ))
}

// ── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }

    fn f64_bytes(values: &[f64]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }

    #[test]
    fn short_circuits_when_both_flags_off() {
        // Even for NaN-bearing input, no scan should run if both flags are off.
        let data = f32_bytes(&[f32::NAN, f32::INFINITY]);
        scan(
            &data,
            Dtype::Float32,
            ByteOrder::native(),
            false,
            false,
            false,
        )
        .expect("both flags off must short-circuit");
    }

    #[test]
    fn integer_dtype_is_zero_cost() {
        // 0xFFFFFFFF would be NaN if interpreted as f32.  Confirm uint32
        // dtype does not get scanned.
        let data: Vec<u8> = vec![0xFF; 64];
        scan(&data, Dtype::Uint32, ByteOrder::native(), true, true, false)
            .expect("integer dtypes must short-circuit");
    }

    #[test]
    fn bitmask_dtype_is_zero_cost() {
        let data: Vec<u8> = vec![0xFF; 16];
        scan(
            &data,
            Dtype::Bitmask,
            ByteOrder::native(),
            true,
            true,
            false,
        )
        .expect("bitmask dtype must short-circuit");
    }

    #[test]
    fn sequential_f32_reports_global_first_nan() {
        let data = f32_bytes(&[1.0, 2.0, 3.0, f32::NAN, 5.0, f32::NAN]);
        let err = scan(
            &data,
            Dtype::Float32,
            ByteOrder::native(),
            true,
            false,
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("element 3"));
    }

    #[test]
    fn sequential_f64_reports_positive_inf_sign() {
        let data = f64_bytes(&[1.0, f64::INFINITY]);
        let err = scan(
            &data,
            Dtype::Float64,
            ByteOrder::native(),
            false,
            true,
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("+Inf"));
    }

    #[test]
    fn sequential_f64_reports_negative_inf_sign() {
        let data = f64_bytes(&[1.0, f64::NEG_INFINITY]);
        let err = scan(
            &data,
            Dtype::Float64,
            ByteOrder::native(),
            false,
            true,
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("-Inf"));
    }

    #[test]
    fn float16_quiet_nan_detected() {
        // 0x7E00 has exp=0x1F, mantissa != 0 → NaN
        let bits: u16 = 0x7E00;
        let data: Vec<u8> = bits.to_ne_bytes().to_vec();
        let err = scan(
            &data,
            Dtype::Float16,
            ByteOrder::native(),
            true,
            false,
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("NaN"));
        assert!(err.to_string().contains("float16"));
    }

    #[test]
    fn float16_inf_detected() {
        // 0x7C00 is +Inf: exp=0x1F, mantissa=0
        let bits: u16 = 0x7C00;
        let data: Vec<u8> = bits.to_ne_bytes().to_vec();
        let err = scan(
            &data,
            Dtype::Float16,
            ByteOrder::native(),
            false,
            true,
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("+Inf"));
    }

    #[test]
    fn bfloat16_quiet_nan_detected() {
        let bits: u16 = 0x7FC0;
        let data: Vec<u8> = bits.to_ne_bytes().to_vec();
        let err = scan(
            &data,
            Dtype::Bfloat16,
            ByteOrder::native(),
            true,
            false,
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("NaN"));
    }

    #[test]
    fn complex64_nan_in_imag_detected() {
        let data: Vec<u8> = [1.0_f32, f32::NAN]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let err = scan(
            &data,
            Dtype::Complex64,
            ByteOrder::native(),
            true,
            false,
            false,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("imaginary"), "got: {msg}");
    }

    #[test]
    fn finite_data_passes_when_both_flags_on() {
        let data = f64_bytes(&[0.0, -0.0, 1.0, -1.0, f64::MIN, f64::MAX]);
        scan(
            &data,
            Dtype::Float64,
            ByteOrder::native(),
            true,
            true,
            false,
        )
        .expect("finite data must pass both flags");
    }

    #[cfg(feature = "threads")]
    #[test]
    fn parallel_path_rejects_nan() {
        // Above PAR_CHUNK_BYTES so the parallel branch is taken.
        let mut values = vec![1.0_f64; 16_384]; // 128 KiB
        values[9_001] = f64::NAN;
        let data = f64_bytes(&values);
        let err = scan(
            &data,
            Dtype::Float64,
            ByteOrder::native(),
            true,
            false,
            true,
        )
        .unwrap_err();
        assert!(err.to_string().contains("NaN"));
    }
}
