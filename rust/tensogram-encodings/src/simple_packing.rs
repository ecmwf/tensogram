// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PackingError {
    #[error("NaN value encountered at index {0}")]
    NanValue(usize),
    #[error("bits_per_value {0} exceeds maximum of 64")]
    BitsPerValueTooLarge(u32),
    #[error("insufficient data: expected at least {expected} bytes, got {actual}")]
    InsufficientData { expected: usize, actual: usize },
    #[error("output size overflow: {num_values} values × {bytes_per_value} bytes")]
    OutputSizeOverflow {
        num_values: usize,
        bytes_per_value: usize,
    },
}

/// Minimum number of values below which the parallel simple_packing
/// paths (min/max scan, encode, decode) fall back to sequential.
/// Rayon's par_iter cost on small buffers exceeds the gain; 64 KiB of
/// f64 = 8192 values is a reasonable break-even on modern CPUs.
#[cfg(feature = "threads")]
const PARALLEL_MIN_VALUES: usize = 8192;

/// Run a min/max scan over `values`, returning `Err(NanValue)` on the
/// first NaN observed (or any NaN when running in parallel).
fn scan_min_max(
    values: &[f64],
    #[allow(unused_variables)] threads: u32,
) -> Result<(f64, f64), PackingError> {
    #[cfg(feature = "threads")]
    {
        if threads >= 2 && values.len() >= PARALLEL_MIN_VALUES {
            use rayon::prelude::*;
            // Short-circuit on NaN: `try_fold` stops as soon as one
            // worker sees a NaN.  The reported index is from whichever
            // chunk that worker owned — not necessarily the globally-
            // first NaN.  This trade-off was accepted for
            // `threads > 0` callers; sequential callers see the
            // original first-NaN behaviour.
            return values
                .par_iter()
                .enumerate()
                .try_fold(
                    || (f64::INFINITY, f64::NEG_INFINITY),
                    |(mn, mx), (i, &v)| {
                        if v.is_nan() {
                            Err(PackingError::NanValue(i))
                        } else {
                            Ok((mn.min(v), mx.max(v)))
                        }
                    },
                )
                .try_reduce(
                    || (f64::INFINITY, f64::NEG_INFINITY),
                    |(amn, amx), (bmn, bmx)| Ok((amn.min(bmn), amx.max(bmx))),
                );
        }
    }

    // Sequential scan — preserves first-NaN-index semantics.
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() {
            return Err(PackingError::NanValue(i));
        }
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }
    Ok((min_val, max_val))
}

#[derive(Debug, Clone)]
pub struct SimplePackingParams {
    pub reference_value: f64,
    pub binary_scale_factor: i32,
    pub decimal_scale_factor: i32,
    pub bits_per_value: u32,
}

pub fn compute_params(
    values: &[f64],
    bits_per_value: u32,
    decimal_scale_factor: i32,
) -> Result<SimplePackingParams, PackingError> {
    compute_params_with_threads(values, bits_per_value, decimal_scale_factor, 0)
}

/// Thread-aware variant of [`compute_params`].
///
/// `threads == 0` preserves the pre-0.13.0 sequential scan, including
/// the guarantee that the reported `NanValue(i)` is the **first** NaN
/// in `values`.
///
/// `threads > 0` splits the scan across rayon workers.  Output
/// `SimplePackingParams` are byte-identical to the sequential path
/// (min/max are associative and NaN-free slices produce the same
/// reduction regardless of split).  The caveat is NaN reporting:
/// when more than one NaN is present, the **index reported** may
/// belong to any of them (rayon's work-stealing reduction is not
/// order-preserving for short-circuit errors).  This trade-off was
/// accepted in the feature design — see
/// `docs/src/guide/multi-threaded-pipeline.md`.
pub fn compute_params_with_threads(
    values: &[f64],
    bits_per_value: u32,
    decimal_scale_factor: i32,
    #[allow(unused_variables)] threads: u32,
) -> Result<SimplePackingParams, PackingError> {
    if bits_per_value > 64 {
        return Err(PackingError::BitsPerValueTooLarge(bits_per_value));
    }

    if values.is_empty() || bits_per_value == 0 {
        let reference_value = values.first().copied().unwrap_or(0.0);
        return Ok(SimplePackingParams {
            reference_value,
            binary_scale_factor: 0,
            decimal_scale_factor,
            bits_per_value,
        });
    }

    // Parallel scan when the `threads` feature is compiled in and the
    // caller requested it AND the input is large enough to amortise
    // the rayon split overhead.
    let (min_val, max_val) = scan_min_max(values, threads)?;

    let d_scale = 10f64.powi(decimal_scale_factor);

    let reference_value = min_val;
    let range = (max_val - min_val) * d_scale;

    let max_packed = if bits_per_value >= 64 {
        u64::MAX as f64
    } else {
        ((1u64 << bits_per_value) - 1) as f64
    };

    // binary_scale_factor E: range = 2^E * max_packed, so E = log2(range / max_packed)
    let binary_scale_factor = if range == 0.0 || max_packed == 0.0 {
        0
    } else {
        (range / max_packed).log2().ceil() as i32
    };

    Ok(SimplePackingParams {
        reference_value,
        binary_scale_factor,
        decimal_scale_factor,
        bits_per_value,
    })
}

/// Encode f64 values to packed integer bytes (MSB-first bit packing).
pub fn encode(values: &[f64], params: &SimplePackingParams) -> Result<Vec<u8>, PackingError> {
    encode_with_threads(values, params, 0)
}

/// Thread-aware variant of [`encode`].
///
/// `threads == 0` preserves the pre-0.13.0 sequential path.
/// `threads > 0` splits the packing work across rayon workers.  Output
/// bytes are byte-identical to the sequential path regardless of
/// thread count (simple_packing is transparent under chunked
/// parallelism — each chunk produces the same MSB-first bits it would
/// in a sequential scan).
///
/// NaN handling: same short-circuit semantics as
/// [`compute_params_with_threads`] — when `threads > 0` the reported
/// NaN index may belong to any of several NaNs in the input.
pub fn encode_with_threads(
    values: &[f64],
    params: &SimplePackingParams,
    #[allow(unused_variables)] threads: u32,
) -> Result<Vec<u8>, PackingError> {
    if params.bits_per_value > 64 {
        return Err(PackingError::BitsPerValueTooLarge(params.bits_per_value));
    }

    // NaN scan.  When parallel, this is fused into the packing loop
    // below; sequential goes through the original fast check so that
    // the first-NaN-index guarantee holds.
    #[cfg(feature = "threads")]
    let parallel = threads >= 2 && values.len() >= PARALLEL_MIN_VALUES;
    #[cfg(not(feature = "threads"))]
    let parallel = false;

    if !parallel {
        if let Some(i) = values.iter().position(|v| v.is_nan()) {
            return Err(PackingError::NanValue(i));
        }
    }

    if params.bits_per_value == 0 {
        return Ok(Vec::new());
    }

    let bpv = params.bits_per_value;
    // packed_int = round((value - ref) * 10^D / 2^E)
    //            = round((value - ref) * scale)
    let scale = 10f64.powi(params.decimal_scale_factor) * 2f64.powi(-params.binary_scale_factor);
    let refv = params.reference_value;
    let max_packed: u64 = if bpv >= 64 {
        u64::MAX
    } else {
        (1u64 << bpv) - 1
    };

    if parallel {
        #[cfg(feature = "threads")]
        {
            return match bpv {
                8 => encode_aligned_par::<1>(values, refv, scale, max_packed),
                16 => encode_aligned_par::<2>(values, refv, scale, max_packed),
                24 => encode_aligned_par::<3>(values, refv, scale, max_packed),
                32 => encode_aligned_par::<4>(values, refv, scale, max_packed),
                _ => encode_generic_par(values, refv, scale, max_packed, bpv),
            };
        }
    }

    match bpv {
        8 => encode_aligned::<1>(values, refv, scale, max_packed),
        16 => encode_aligned::<2>(values, refv, scale, max_packed),
        24 => encode_aligned::<3>(values, refv, scale, max_packed),
        32 => encode_aligned::<4>(values, refv, scale, max_packed),
        _ => encode_generic(values, refv, scale, max_packed, bpv),
    }
}

/// Write `q` into a chunk as `N` MSB-first big-endian bytes.
///
/// Shared by the aligned sequential and parallel paths.  `N` is always
/// one of 1, 2, 3, 4 — the compiler specialises each caller.
#[inline]
fn splat_aligned<const N: usize>(chunk: &mut [u8], q: u64) {
    match N {
        1 => {
            chunk[0] = q as u8;
        }
        2 => {
            chunk[0] = (q >> 8) as u8;
            chunk[1] = q as u8;
        }
        3 => {
            chunk[0] = (q >> 16) as u8;
            chunk[1] = (q >> 8) as u8;
            chunk[2] = q as u8;
        }
        4 => {
            chunk[0] = (q >> 24) as u8;
            chunk[1] = (q >> 16) as u8;
            chunk[2] = (q >> 8) as u8;
            chunk[3] = q as u8;
        }
        _ => unreachable!("encode_aligned only instantiated for N in 1..=4"),
    }
}

fn encode_aligned<const N: usize>(
    values: &[f64],
    refv: f64,
    scale: f64,
    max_packed: u64,
) -> Result<Vec<u8>, PackingError> {
    let len = values
        .len()
        .checked_mul(N)
        .ok_or(PackingError::OutputSizeOverflow {
            num_values: values.len(),
            bytes_per_value: N,
        })?;
    let mut out = vec![0u8; len];

    for (chunk, &v) in out.chunks_exact_mut(N).zip(values) {
        // Saturating f64→u64 cast handles negative values (→ 0).
        // u64::min handles the rare +1 overshoot from rounding.
        let q = (((v - refv) * scale).round() as u64).min(max_packed);
        splat_aligned::<N>(chunk, q);
    }
    Ok(out)
}

/// Parallel aligned encode.  Each output N-byte chunk is computed
/// independently from the corresponding input f64, so rayon's
/// `par_chunks_mut` gives byte-identical output to the sequential
/// path.  NaN check is fused into the packing loop and short-circuits
/// the whole parallel reduction on the first sighting in the current
/// worker's slice.
#[cfg(feature = "threads")]
fn encode_aligned_par<const N: usize>(
    values: &[f64],
    refv: f64,
    scale: f64,
    max_packed: u64,
) -> Result<Vec<u8>, PackingError> {
    use rayon::prelude::*;

    let len = values
        .len()
        .checked_mul(N)
        .ok_or(PackingError::OutputSizeOverflow {
            num_values: values.len(),
            bytes_per_value: N,
        })?;
    let mut out = vec![0u8; len];

    // Work per chunk is ~N bytes + 1 f64 load.  We let rayon pick
    // chunk boundaries via par_chunks_mut; pair output chunks with
    // input values by index.
    out.par_chunks_exact_mut(N)
        .zip(values.par_iter().enumerate())
        .try_for_each(|(chunk, (i, &v))| -> Result<(), PackingError> {
            if v.is_nan() {
                return Err(PackingError::NanValue(i));
            }
            let q = (((v - refv) * scale).round() as u64).min(max_packed);
            splat_aligned::<N>(chunk, q);
            Ok(())
        })?;

    Ok(out)
}

/// Parallel generic encode for non-byte-aligned bit widths.
///
/// Splits the work into chunks sized at `lcm(8, bpv) / bpv` values —
/// guaranteeing each chunk starts and ends on a byte boundary, so
/// workers never share output bytes.  Output bytes are byte-identical
/// to the sequential path.
#[cfg(feature = "threads")]
fn encode_generic_par(
    values: &[f64],
    refv: f64,
    scale: f64,
    max_packed: u64,
    bpv: u32,
) -> Result<Vec<u8>, PackingError> {
    use rayon::prelude::*;

    let total_bits =
        (values.len() as u64)
            .checked_mul(bpv as u64)
            .ok_or(PackingError::OutputSizeOverflow {
                num_values: values.len(),
                bytes_per_value: (bpv as usize).div_ceil(8),
            })?;
    let total_bytes =
        usize::try_from(total_bits.div_ceil(8)).map_err(|_| PackingError::OutputSizeOverflow {
            num_values: values.len(),
            bytes_per_value: (bpv as usize).div_ceil(8),
        })?;
    let mut output = vec![0u8; total_bytes];

    // Chunk size in values: lcm(8, bpv) / bpv.
    // lcm(a, b) = a*b / gcd(a, b).
    let bpv_usize = bpv as usize;
    let g = gcd(8, bpv_usize);
    let values_per_chunk = (8 * bpv_usize / g) / bpv_usize;
    let bytes_per_chunk = values_per_chunk * bpv_usize / 8;

    // Pair value chunks with output byte chunks.  The last chunk may
    // be shorter — fall back to sequential packing for it so we
    // don't need a separate tail path.
    let full_chunks = values.len() / values_per_chunk;
    let head_values = &values[..full_chunks * values_per_chunk];
    let head_bytes = &mut output[..full_chunks * bytes_per_chunk];
    head_values
        .par_chunks(values_per_chunk)
        .zip(head_bytes.par_chunks_mut(bytes_per_chunk))
        .try_for_each(|(vchunk, bchunk)| -> Result<(), PackingError> {
            let mut bit_pos: u64 = 0;
            for (local_i, &value) in vchunk.iter().enumerate() {
                if value.is_nan() {
                    // Index is chunk-local — we cannot easily recover
                    // the global one under par without extra bookkeeping.
                    // The caller explicitly opted into threads=N; doc'd
                    // non-determinism of reported NaN index is accepted.
                    return Err(PackingError::NanValue(local_i));
                }
                let packed = (((value - refv) * scale).round() as u64).min(max_packed);
                write_bits(bchunk, bit_pos, packed, bpv);
                bit_pos += bpv as u64;
            }
            Ok(())
        })?;

    // Tail (if any): sequential.
    let tail_values = &values[full_chunks * values_per_chunk..];
    if !tail_values.is_empty() {
        let tail_byte_start = full_chunks * bytes_per_chunk;
        let tail_bytes = &mut output[tail_byte_start..];
        let mut bit_pos: u64 = 0;
        for (i, &value) in tail_values.iter().enumerate() {
            if value.is_nan() {
                return Err(PackingError::NanValue(full_chunks * values_per_chunk + i));
            }
            let packed = (((value - refv) * scale).round() as u64).min(max_packed);
            write_bits(tail_bytes, bit_pos, packed, bpv);
            bit_pos += bpv as u64;
        }
    }

    Ok(output)
}

#[cfg(feature = "threads")]
const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn encode_generic(
    values: &[f64],
    refv: f64,
    scale: f64,
    max_packed: u64,
    bpv: u32,
) -> Result<Vec<u8>, PackingError> {
    let total_bits =
        (values.len() as u64)
            .checked_mul(bpv as u64)
            .ok_or(PackingError::OutputSizeOverflow {
                num_values: values.len(),
                bytes_per_value: (bpv as usize).div_ceil(8),
            })?;
    let total_bytes =
        usize::try_from(total_bits.div_ceil(8)).map_err(|_| PackingError::OutputSizeOverflow {
            num_values: values.len(),
            bytes_per_value: (bpv as usize).div_ceil(8),
        })?;
    let mut output = vec![0u8; total_bytes];

    let mut bit_pos: u64 = 0;
    for &value in values {
        let packed = (((value - refv) * scale).round() as u64).min(max_packed);
        write_bits(&mut output, bit_pos, packed, bpv);
        bit_pos += bpv as u64;
    }

    Ok(output)
}

/// Decode packed bytes back to f64 values.
pub fn decode(
    packed: &[u8],
    num_values: usize,
    params: &SimplePackingParams,
) -> Result<Vec<f64>, PackingError> {
    decode_with_threads(packed, num_values, params, 0)
}

/// Thread-aware variant of [`decode`].
///
/// `threads == 0` preserves the pre-0.13.0 sequential path.
/// `threads > 0` uses rayon to unpack in parallel.  Output values are
/// byte-identical to the sequential path (floating-point math is
/// trivially associative here — each value is computed independently
/// from its packed bits).
pub fn decode_with_threads(
    packed: &[u8],
    num_values: usize,
    params: &SimplePackingParams,
    #[allow(unused_variables)] threads: u32,
) -> Result<Vec<f64>, PackingError> {
    if params.bits_per_value > 64 {
        return Err(PackingError::BitsPerValueTooLarge(params.bits_per_value));
    }

    if params.bits_per_value == 0 {
        return Ok(vec![params.reference_value; num_values]);
    }

    let bpv = params.bits_per_value;
    let total_bits = num_values as u64 * bpv as u64;
    let required_bytes = total_bits.div_ceil(8) as usize;
    if packed.len() < required_bytes {
        return Err(PackingError::InsufficientData {
            expected: required_bytes,
            actual: packed.len(),
        });
    }

    let refv = params.reference_value;
    // value = ref + 2^E * 10^(-D) * packed_int = ref + inv_scale * packed_int
    let inv_scale =
        2f64.powi(params.binary_scale_factor) * 10f64.powi(-params.decimal_scale_factor);

    #[cfg(feature = "threads")]
    let parallel = threads >= 2 && num_values >= PARALLEL_MIN_VALUES;
    #[cfg(not(feature = "threads"))]
    let parallel = false;

    if parallel {
        #[cfg(feature = "threads")]
        {
            return Ok(match bpv {
                8 => decode_aligned_par::<1>(packed, num_values, refv, inv_scale),
                16 => decode_aligned_par::<2>(packed, num_values, refv, inv_scale),
                24 => decode_aligned_par::<3>(packed, num_values, refv, inv_scale),
                32 => decode_aligned_par::<4>(packed, num_values, refv, inv_scale),
                _ => decode_generic_par(packed, num_values, refv, inv_scale, bpv),
            });
        }
    }

    match bpv {
        8 => Ok(decode_aligned::<1>(packed, num_values, refv, inv_scale)),
        16 => Ok(decode_aligned::<2>(packed, num_values, refv, inv_scale)),
        24 => Ok(decode_aligned::<3>(packed, num_values, refv, inv_scale)),
        32 => Ok(decode_aligned::<4>(packed, num_values, refv, inv_scale)),
        _ => Ok(decode_generic(packed, num_values, refv, inv_scale, bpv)),
    }
}

/// Read `N` MSB-first big-endian bytes from `chunk` as a `u64`.
///
/// Shared by the aligned sequential and parallel decode paths.
#[inline]
fn gather_aligned<const N: usize>(chunk: &[u8]) -> u64 {
    match N {
        1 => chunk[0] as u64,
        2 => ((chunk[0] as u64) << 8) | chunk[1] as u64,
        3 => ((chunk[0] as u64) << 16) | ((chunk[1] as u64) << 8) | chunk[2] as u64,
        4 => {
            ((chunk[0] as u64) << 24)
                | ((chunk[1] as u64) << 16)
                | ((chunk[2] as u64) << 8)
                | chunk[3] as u64
        }
        _ => unreachable!("decode_aligned only instantiated for N in 1..=4"),
    }
}

fn decode_aligned<const N: usize>(
    packed: &[u8],
    num_values: usize,
    refv: f64,
    inv_scale: f64,
) -> Vec<f64> {
    let mut values = Vec::with_capacity(num_values);
    for chunk in packed[..num_values * N].chunks_exact(N) {
        let packed_int = gather_aligned::<N>(chunk);
        values.push(refv + inv_scale * packed_int as f64);
    }
    values
}

/// Parallel aligned decode — byte-identical to the sequential path.
#[cfg(feature = "threads")]
fn decode_aligned_par<const N: usize>(
    packed: &[u8],
    num_values: usize,
    refv: f64,
    inv_scale: f64,
) -> Vec<f64> {
    use rayon::prelude::*;

    let mut values = vec![0.0f64; num_values];
    values
        .par_iter_mut()
        .zip(packed[..num_values * N].par_chunks_exact(N))
        .for_each(|(out, chunk)| {
            let packed_int = gather_aligned::<N>(chunk);
            *out = refv + inv_scale * packed_int as f64;
        });
    values
}

/// Parallel generic decode for non-byte-aligned bit widths.  Uses the
/// same byte-aligned chunking trick as `encode_generic_par`: chunks
/// of `lcm(8, bpv) / bpv` values start and end on byte boundaries.
#[cfg(feature = "threads")]
fn decode_generic_par(
    packed: &[u8],
    num_values: usize,
    refv: f64,
    inv_scale: f64,
    bpv: u32,
) -> Vec<f64> {
    use rayon::prelude::*;

    let bpv_usize = bpv as usize;
    let g = gcd(8, bpv_usize);
    let values_per_chunk = (8 * bpv_usize / g) / bpv_usize;
    let bytes_per_chunk = values_per_chunk * bpv_usize / 8;

    let full_chunks = num_values / values_per_chunk;
    let mut values = vec![0.0f64; num_values];

    {
        let head_values_slice = &mut values[..full_chunks * values_per_chunk];
        let head_bytes = &packed[..full_chunks * bytes_per_chunk];

        head_values_slice
            .par_chunks_mut(values_per_chunk)
            .zip(head_bytes.par_chunks(bytes_per_chunk))
            .for_each(|(vchunk, bchunk)| {
                let mut bit_pos: u64 = 0;
                for out in vchunk.iter_mut() {
                    let packed_int = read_bits(bchunk, bit_pos, bpv);
                    *out = refv + inv_scale * packed_int as f64;
                    bit_pos += bpv as u64;
                }
            });
    }

    // Tail — sequential.
    let tail_count = num_values - full_chunks * values_per_chunk;
    if tail_count > 0 {
        let tail_byte_start = full_chunks * bytes_per_chunk;
        let tail_bytes = &packed[tail_byte_start..];
        let tail_values = &mut values[full_chunks * values_per_chunk..];
        let mut bit_pos: u64 = 0;
        for out in tail_values.iter_mut() {
            let packed_int = read_bits(tail_bytes, bit_pos, bpv);
            *out = refv + inv_scale * packed_int as f64;
            bit_pos += bpv as u64;
        }
    }

    values
}

fn decode_generic(
    packed: &[u8],
    num_values: usize,
    refv: f64,
    inv_scale: f64,
    bpv: u32,
) -> Vec<f64> {
    let mut values = Vec::with_capacity(num_values);
    let mut bit_pos: u64 = 0;

    for _ in 0..num_values {
        let packed_int = read_bits(packed, bit_pos, bpv);
        values.push(refv + inv_scale * packed_int as f64);
        bit_pos += bpv as u64;
    }
    values
}

/// Decode a range of packed values starting at an arbitrary bit offset.
///
/// `packed` is a byte slice containing the packed bits.
/// `bit_offset` is the starting bit position within `packed`.
/// `num_values` is how many values to unpack.
pub fn decode_range(
    packed: &[u8],
    bit_offset: usize,
    num_values: usize,
    params: &SimplePackingParams,
) -> Result<Vec<f64>, PackingError> {
    if params.bits_per_value > 64 {
        return Err(PackingError::BitsPerValueTooLarge(params.bits_per_value));
    }

    if params.bits_per_value == 0 {
        return Ok(vec![params.reference_value; num_values]);
    }

    let bpv = params.bits_per_value;

    let end_bit = bit_offset as u64 + num_values as u64 * bpv as u64;
    let required_bytes = end_bit.div_ceil(8) as usize;
    if packed.len() < required_bytes {
        return Err(PackingError::InsufficientData {
            expected: required_bytes,
            actual: packed.len(),
        });
    }

    let d_factor = 10f64.powi(-params.decimal_scale_factor);
    let e_factor = 2f64.powi(params.binary_scale_factor);

    let mut values = Vec::with_capacity(num_values);
    let mut bit_pos = bit_offset as u64;

    for _ in 0..num_values {
        let packed_int = read_bits(packed, bit_pos, bpv);
        let value = params.reference_value + e_factor * d_factor * packed_int as f64;
        values.push(value);
        bit_pos += bpv as u64;
    }

    Ok(values)
}

/// Write `nbits` bits of `value` at `bit_offset` in `buf`, MSB-first.
fn write_bits(buf: &mut [u8], bit_offset: u64, value: u64, nbits: u32) {
    if nbits == 0 {
        return;
    }

    if bit_offset.is_multiple_of(8) {
        let idx = (bit_offset / 8) as usize;
        match nbits {
            8 => {
                buf[idx] = value as u8;
                return;
            }
            16 => {
                buf[idx] = (value >> 8) as u8;
                buf[idx + 1] = value as u8;
                return;
            }
            24 => {
                buf[idx] = (value >> 16) as u8;
                buf[idx + 1] = (value >> 8) as u8;
                buf[idx + 2] = value as u8;
                return;
            }
            32 => {
                buf[idx] = (value >> 24) as u8;
                buf[idx + 1] = (value >> 16) as u8;
                buf[idx + 2] = (value >> 8) as u8;
                buf[idx + 3] = value as u8;
                return;
            }
            _ => {}
        }
    }

    let mut remaining = nbits;
    let mut pos = bit_offset as usize;
    let mut val = value;

    let first_avail = 8 - (pos % 8);
    if remaining <= first_avail as u32 {
        // Entire value fits in one byte
        let byte_idx = pos / 8;
        let shift = first_avail as u32 - remaining;
        let mask = ((1u64 << remaining) - 1) as u8;
        buf[byte_idx] |= (val as u8 & mask) << shift;
        return;
    }

    // Write partial first byte
    if !pos.is_multiple_of(8) {
        let bits_in_first = first_avail as u32;
        let byte_idx = pos / 8;
        let top_bits = (val >> (remaining - bits_in_first)) as u8;
        let mask = (1u8 << bits_in_first) - 1;
        buf[byte_idx] |= top_bits & mask;
        remaining -= bits_in_first;
        pos += bits_in_first as usize;
        val &= (1u64 << remaining) - 1;
    }

    // Write full bytes
    while remaining >= 8 {
        remaining -= 8;
        buf[pos / 8] = (val >> remaining) as u8;
        pos += 8;
        if remaining > 0 {
            val &= (1u64 << remaining) - 1;
        }
    }

    // Write partial last byte
    if remaining > 0 {
        let shift = 8 - remaining;
        buf[pos / 8] |= (val as u8) << shift;
    }
}

/// Read `nbits` bits at `bit_offset` from `buf`, MSB-first.
fn read_bits(buf: &[u8], bit_offset: u64, nbits: u32) -> u64 {
    if nbits == 0 {
        return 0;
    }
    let mut remaining = nbits;
    let mut pos = bit_offset as usize;
    let mut value: u64 = 0;

    // Bits available in the first byte
    let first_avail = (8 - (pos % 8)) as u32;
    if remaining <= first_avail {
        // Entire value in one byte
        let byte_idx = pos / 8;
        let shift = first_avail - remaining;
        let mask = (1u64 << remaining) - 1;
        return ((buf[byte_idx] >> shift) as u64) & mask;
    }

    // Read partial first byte
    if !pos.is_multiple_of(8) {
        let bits_in_first = first_avail;
        let byte_idx = pos / 8;
        let mask = (1u8 << bits_in_first) - 1;
        value = (buf[byte_idx] & mask) as u64;
        remaining -= bits_in_first;
        pos += bits_in_first as usize;
    }

    // Read full bytes
    while remaining >= 8 {
        value = (value << 8) | buf[pos / 8] as u64;
        remaining -= 8;
        pos += 8;
    }

    // Read partial last byte
    if remaining > 0 {
        let shift = 8 - remaining;
        value = (value << remaining) | ((buf[pos / 8] >> shift) as u64);
    }

    value
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Threading determinism ─────────────────────────────────────────
    //
    // The invariants tested here are the foundation of the
    // multi-threaded pipeline: for simple_packing the output must be
    // byte-identical across all thread counts (it is a transparent
    // codec — no internal block reordering).  This is stronger than
    // the blosc2/zstd contract which only guarantees round-trip
    // identity.

    #[cfg(feature = "threads")]
    #[test]
    fn threads_byte_identical_aligned_widths() {
        let values: Vec<f64> = (0..100_000).map(|i| 200.0 + i as f64 * 0.001).collect();
        for bpv in [8u32, 16, 24, 32] {
            let params = compute_params(&values, bpv, 0).unwrap();
            let seq = encode_with_threads(&values, &params, 0).unwrap();
            for t in [1u32, 2, 4, 8] {
                let par = encode_with_threads(&values, &params, t).unwrap();
                assert_eq!(seq, par, "encode threads={t} bpv={bpv} mismatch");
                let rt = decode_with_threads(&par, values.len(), &params, t).unwrap();
                let rt_seq = decode_with_threads(&seq, values.len(), &params, 0).unwrap();
                assert_eq!(rt, rt_seq, "decode threads={t} bpv={bpv} mismatch");
            }
        }
    }

    #[cfg(feature = "threads")]
    #[test]
    fn threads_byte_identical_generic_widths() {
        // Non-byte-aligned widths exercise the chunked-lcm parallel path.
        let values: Vec<f64> = (0..100_000).map(|i| 1.0 + i as f64 * 0.001).collect();
        for bpv in [12u32, 20, 7, 13] {
            let params = compute_params(&values, bpv, 0).unwrap();
            let seq = encode_with_threads(&values, &params, 0).unwrap();
            for t in [1u32, 2, 4, 8] {
                let par = encode_with_threads(&values, &params, t).unwrap();
                assert_eq!(seq, par, "encode threads={t} bpv={bpv} mismatch");
                let rt = decode_with_threads(&par, values.len(), &params, t).unwrap();
                let rt_seq = decode_with_threads(&seq, values.len(), &params, 0).unwrap();
                assert_eq!(rt, rt_seq, "decode threads={t} bpv={bpv} mismatch");
            }
        }
    }

    #[cfg(feature = "threads")]
    #[test]
    fn threads_below_threshold_uses_sequential_path() {
        // Below PARALLEL_MIN_VALUES the threaded API should still
        // produce byte-identical output (it falls back to sequential).
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let params = compute_params(&values, 16, 0).unwrap();
        let seq = encode(&values, &params).unwrap();
        let par = encode_with_threads(&values, &params, 8).unwrap();
        assert_eq!(seq, par);
    }

    #[cfg(feature = "threads")]
    #[test]
    fn threads_compute_params_matches_sequential_on_clean_input() {
        // Without NaN, the min/max reduction produces identical params.
        let values: Vec<f64> = (0..100_000)
            .map(|i| (i as f64).sin() * 50.0 + 100.0)
            .collect();
        let seq = compute_params(&values, 16, 0).unwrap();
        let par = compute_params_with_threads(&values, 16, 0, 8).unwrap();
        assert_eq!(seq.reference_value.to_bits(), par.reference_value.to_bits());
        assert_eq!(seq.binary_scale_factor, par.binary_scale_factor);
        assert_eq!(seq.bits_per_value, par.bits_per_value);
    }

    #[test]
    fn test_constant_field() {
        let values = vec![42.0; 100];
        let params = SimplePackingParams {
            reference_value: 42.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 0,
        };
        let encoded = encode(&values, &params).unwrap();
        assert!(encoded.is_empty());
        let decoded = decode(&encoded, 100, &params).unwrap();
        assert_eq!(decoded.len(), 100);
        for v in decoded {
            assert!((v - 42.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_nan_rejection_in_compute_params() {
        let values = vec![1.0, f64::NAN, 3.0];
        assert!(matches!(
            compute_params(&values, 16, 0),
            Err(PackingError::NanValue(1))
        ));
    }

    #[test]
    fn test_nan_rejection_in_encode() {
        let values = vec![1.0, f64::NAN, 3.0];
        let params = SimplePackingParams {
            reference_value: 0.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 16,
        };
        assert!(matches!(
            encode(&values, &params),
            Err(PackingError::NanValue(1))
        ));
    }

    #[test]
    fn test_round_trip_16bit() {
        let values: Vec<f64> = (0..100).map(|i| 230.0 + i as f64 * 0.5).collect();
        let params = compute_params(&values, 16, 0).unwrap();
        let encoded = encode(&values, &params).unwrap();
        let decoded = decode(&encoded, values.len(), &params).unwrap();
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.01, "orig={orig}, dec={dec}");
        }
    }

    #[test]
    fn test_round_trip_12bit() {
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let params = compute_params(&values, 12, 0).unwrap();
        let encoded = encode(&values, &params).unwrap();
        // 10 values * 12 bits = 120 bits = 15 bytes
        assert_eq!(encoded.len(), 15);
        let decoded = decode(&encoded, values.len(), &params).unwrap();
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.01, "orig={orig}, dec={dec}");
        }
    }

    #[test]
    fn test_round_trip_1bit() {
        let values = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let params = SimplePackingParams {
            reference_value: 0.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 1,
        };
        let encoded = encode(&values, &params).unwrap();
        // 9 bits = 2 bytes
        assert_eq!(encoded.len(), 2);
        let decoded = decode(&encoded, values.len(), &params).unwrap();
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bits_per_value_too_large() {
        let values = vec![1.0];
        let params = SimplePackingParams {
            reference_value: 0.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 65,
        };
        assert!(encode(&values, &params).is_err());
    }

    // ── Coverage: edge cases ─────────────────────────────────────────

    #[test]
    fn test_compute_params_nan_detection() {
        let values = vec![1.0, f64::NAN, 3.0];
        let result = compute_params(&values, 16, 0);
        assert!(result.is_err());
        match result.unwrap_err() {
            PackingError::NanValue(idx) => assert_eq!(idx, 1),
            other => panic!("expected NanValue, got {other:?}"),
        }
    }

    #[test]
    fn test_compute_params_nan_at_start() {
        let values = vec![f64::NAN, 1.0, 2.0];
        match compute_params(&values, 16, 0).unwrap_err() {
            PackingError::NanValue(idx) => assert_eq!(idx, 0),
            other => panic!("expected NanValue(0), got {other:?}"),
        }
    }

    #[test]
    fn test_compute_params_empty_array() {
        let params = compute_params(&[], 8, 0).unwrap();
        assert_eq!(params.reference_value, 0.0);
        assert_eq!(params.binary_scale_factor, 0);
    }

    #[test]
    fn test_compute_params_zero_bits() {
        let params = compute_params(&[1.5, 2.5], 0, 0).unwrap();
        assert_eq!(params.bits_per_value, 0);
        assert_eq!(params.reference_value, 1.5);
    }

    #[test]
    fn test_compute_params_bpv_too_large() {
        match compute_params(&[1.0], 65, 0).unwrap_err() {
            PackingError::BitsPerValueTooLarge(bpv) => assert_eq!(bpv, 65),
            other => panic!("expected BitsPerValueTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn test_compute_params_constant_field() {
        // All values identical → range = 0
        let params = compute_params(&[42.0; 100], 16, 0).unwrap();
        assert_eq!(params.binary_scale_factor, 0);
        assert!((params.reference_value - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_params_64_bit() {
        // bits_per_value = 64 → uses u64::MAX branch
        let params = compute_params(&[0.0, 1e18], 64, 0).unwrap();
        assert_eq!(params.bits_per_value, 64);
    }

    #[test]
    fn test_encode_zero_bits() {
        let params = SimplePackingParams {
            reference_value: 5.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 0,
        };
        let encoded = encode(&[5.0, 5.0, 5.0], &params).unwrap();
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_decode_zero_bits() {
        let params = SimplePackingParams {
            reference_value: 7.5,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 0,
        };
        let decoded = decode(&[], 5, &params).unwrap();
        assert_eq!(decoded.len(), 5);
        for v in &decoded {
            assert!((v - 7.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_decode_insufficient_data() {
        let params = SimplePackingParams {
            reference_value: 0.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 8,
        };
        // Need 100 bytes for 100 values at 8 bpv, provide only 10
        match decode(&[0u8; 10], 100, &params).unwrap_err() {
            PackingError::InsufficientData { expected, actual } => {
                assert_eq!(expected, 100);
                assert_eq!(actual, 10);
            }
            other => panic!("expected InsufficientData, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_range_zero_bits() {
        let params = SimplePackingParams {
            reference_value: 3.125,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 0,
        };
        // decode_range(packed, bit_offset, num_values, params)
        let decoded = decode_range(&[], 0, 3, &params).unwrap();
        assert_eq!(decoded.len(), 3);
        for v in &decoded {
            assert!((v - 3.125).abs() < 1e-10);
        }
    }

    #[test]
    fn test_decode_range_insufficient_data() {
        let params = SimplePackingParams {
            reference_value: 0.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 16,
        };
        // decode_range(packed, bit_offset, num_values, params)
        // bit_offset=80, num_values=3, bpv=16 → need (80+48)/8 = 16 bytes, provide 8
        match decode_range(&[0u8; 8], 80, 3, &params).unwrap_err() {
            PackingError::InsufficientData { .. } => {}
            other => panic!("expected InsufficientData, got {other:?}"),
        }
    }

    #[test]
    fn test_encode_bpv_too_large() {
        let params = SimplePackingParams {
            reference_value: 0.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 65,
        };
        assert!(encode(&[1.0], &params).is_err());
    }

    #[test]
    fn test_decode_bpv_too_large() {
        let params = SimplePackingParams {
            reference_value: 0.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 65,
        };
        assert!(decode(&[0u8; 16], 1, &params).is_err());
    }

    #[test]
    fn test_decode_range_bpv_too_large() {
        let params = SimplePackingParams {
            reference_value: 0.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 65,
        };
        assert!(decode_range(&[0u8; 16], 0, 1, &params).is_err());
    }

    #[test]
    fn test_write_read_bits_zero() {
        // write_bits and read_bits with 0 bits should be no-ops
        let mut buf = vec![0u8; 4];
        write_bits(&mut buf, 0, 0, 0);
        assert_eq!(read_bits(&buf, 0, 0), 0);
    }
}
