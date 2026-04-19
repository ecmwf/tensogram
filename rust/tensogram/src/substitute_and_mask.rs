// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Pre-pipeline non-finite substitution + bitmask assembly.
//!
//! Runs before the encoding pipeline on raw float payloads.  Two
//! behavioural modes:
//!
//! - `allow_nan` / `allow_inf` both `false` — the first NaN / ±Inf is
//!   reported as a [`TensogramError::Encoding`], no allocation or
//!   copy.  This is the default encode policy (see
//!   `plans/BITMASK_FRAME.md` §2).
//!
//! - either / both flags `true` — for each allowed kind, the
//!   corresponding bit in a per-kind [`Vec<bool>`] is set and the
//!   float bytes at that position are overwritten with the
//!   dtype-specific canonical zero (`0.0` for scalars,
//!   `0.0 + 0.0i` for complex).  Disallowed non-finite kinds still
//!   error.
//!
//! The output buffer is returned as [`Cow::Borrowed`] when no
//! substitution occurred (zero-alloc fast path, e.g. finite-only
//! input), and as [`Cow::Owned`] when at least one element was
//! rewritten.  Non-float dtypes always take the borrowed path.
//!
//! This module supersedes the pre-0.17 `finite_check` scanner — the
//! reject-only mode (`allow_nan == false && allow_inf == false`) is
//! functionally equivalent, and the allow-mode is the new
//! bitmask-companion path.
//!
//! ## Complex priority
//!
//! For `complex64` / `complex128`, a single element (the (real, imag)
//! pair) contributes to **at most one** mask, following the priority
//! rule from `plans/BITMASK_FRAME.md` §4:
//!
//! 1. NaN wins over any Inf.
//! 2. +Inf wins over -Inf.
//! 3. Otherwise the element is finite.
//!
//! On substitution both components are written as `0.0`.  On decode
//! (see Commit 6) the canonical bit pattern of the kind is restored
//! to **both** components — bit-exact NaN payloads and mixed
//! real/imag kinds are not preserved.  This is a documented lossy
//! trade-off per §7.1.
//!
//! ## Parallelism
//!
//! `PAR_CHUNK_BYTES = 64 KiB` chunks via rayon when `parallel == true`
//! and the payload is at least one chunk.  Each worker owns its local
//! `MaskSet` slice; the final [`MaskSet`] is assembled by
//! concatenating the per-chunk results in input order so bit
//! positions are globally consistent.
//!
//! Disallowed-kind errors from parallel workers report the first
//! offence that the worker holding that chunk saw — not necessarily
//! the globally first, matching the determinism contract used by the
//! rest of the multi-threaded pipeline.

use std::borrow::Cow;

use crate::dtype::Dtype;
use crate::error::{Result, TensogramError};
use tensogram_encodings::ByteOrder;

/// Parallel chunk size in bytes.  A multiple of 16, 8, 4, 2, 1 so
/// the chunk boundary never splits an element for any supported
/// dtype, and small enough that a single worker finishes a chunk
/// quickly — giving early error-exit a tight latency bound.
const PAR_CHUNK_BYTES: usize = 64 * 1024;

/// Three-kind set of raw (uncompressed) bitmasks produced by
/// [`substitute_and_mask`].
///
/// Each field is `Some(mask)` when at least one element of the
/// corresponding kind was seen, and `None` otherwise.  A mask's
/// length always equals the input's element count, so position `i`
/// in the mask maps 1:1 to the `i`th element of the payload.
///
/// Assembly to the compressed on-wire form happens later, at frame
/// emission time (see [`crate::encode`]).
#[derive(Debug, Clone, Default)]
pub struct MaskSet {
    /// NaN mask — `true` at positions where the original value was
    /// NaN (for complex dtypes: where either component was NaN).
    pub nan: Option<Vec<bool>>,
    /// +Inf mask — `true` at positions where the original value was
    /// `+∞` and neither component was NaN.
    pub pos_inf: Option<Vec<bool>>,
    /// -Inf mask — `true` at positions where the original value was
    /// `−∞` and neither component was NaN or `+∞`.
    pub neg_inf: Option<Vec<bool>>,
    /// Element count of the scan — used to size the bitmasks
    /// consistently when one or more kinds was never seen.
    pub n_elements: usize,
}

impl MaskSet {
    /// Fresh empty set for `n_elements` elements.
    pub fn empty(n_elements: usize) -> Self {
        Self {
            nan: None,
            pos_inf: None,
            neg_inf: None,
            n_elements,
        }
    }

    /// `true` when no kind was observed — the caller can emit a
    /// legacy-compatible frame with no `masks` sub-map.
    pub fn is_empty(&self) -> bool {
        self.nan.is_none() && self.pos_inf.is_none() && self.neg_inf.is_none()
    }

    /// Set bit `index` in the nan mask, lazily allocating the
    /// `Vec<bool>` on first hit.  `n_elements` must already be
    /// populated.
    fn set_nan(&mut self, index: usize) {
        let m = self.nan.get_or_insert_with(|| vec![false; self.n_elements]);
        m[index] = true;
    }
    fn set_pos_inf(&mut self, index: usize) {
        let m = self
            .pos_inf
            .get_or_insert_with(|| vec![false; self.n_elements]);
        m[index] = true;
    }
    fn set_neg_inf(&mut self, index: usize) {
        let m = self
            .neg_inf
            .get_or_insert_with(|| vec![false; self.n_elements]);
        m[index] = true;
    }

    /// Merge a chunk-local set into `self` at element offset `base`.
    /// Used by the parallel assembly path.  `self.n_elements` must
    /// already cover the chunk.
    #[cfg(feature = "threads")]
    fn merge_chunk(&mut self, other: MaskSet, base: usize) {
        if let Some(m) = other.nan {
            for (i, &b) in m.iter().enumerate() {
                if b {
                    self.set_nan(base + i);
                }
            }
        }
        if let Some(m) = other.pos_inf {
            for (i, &b) in m.iter().enumerate() {
                if b {
                    self.set_pos_inf(base + i);
                }
            }
        }
        if let Some(m) = other.neg_inf {
            for (i, &b) in m.iter().enumerate() {
                if b {
                    self.set_neg_inf(base + i);
                }
            }
        }
    }
}

/// Run the substitute-and-mask stage on `data`.
///
/// Returns `(Cow<[u8]>, MaskSet)`:
/// - [`Cow::Borrowed`] when the dtype is non-float (zero-cost
///   passthrough) or when no substitution was necessary (finite-only
///   input in float dtypes).
/// - [`Cow::Owned`] otherwise — a byte copy of `data` with non-finite
///   positions overwritten to the dtype-specific zero.
///
/// # Errors
///
/// [`TensogramError::Encoding`] when the input contains a non-finite
/// kind that the matching `allow_*` flag is `false` for.  The error
/// message names the kind (`"NaN"` / `"+Inf"` / `"-Inf"`), the
/// element index, the dtype, and — for complex types — the component
/// that triggered the rejection.  No partial masks or output buffers
/// are produced on error; the caller's data remains untouched.
pub(crate) fn substitute_and_mask<'a>(
    data: &'a [u8],
    dtype: Dtype,
    byte_order: ByteOrder,
    allow_nan: bool,
    allow_inf: bool,
    parallel: bool,
) -> Result<(Cow<'a, [u8]>, MaskSet)> {
    let elem_size = match dtype {
        Dtype::Float16 | Dtype::Bfloat16 => 2,
        Dtype::Float32 => 4,
        Dtype::Float64 | Dtype::Complex64 => 8,
        Dtype::Complex128 => 16,
        // Non-float dtypes are always finite; skip the scan entirely.
        Dtype::Int8
        | Dtype::Int16
        | Dtype::Int32
        | Dtype::Int64
        | Dtype::Uint8
        | Dtype::Uint16
        | Dtype::Uint32
        | Dtype::Uint64
        | Dtype::Bitmask => {
            return Ok((Cow::Borrowed(data), MaskSet::empty(0)));
        }
    };

    if data.is_empty() {
        return Ok((Cow::Borrowed(data), MaskSet::empty(0)));
    }

    let n_elements = data.len() / elem_size;

    // Parallel dispatch: chunks must be element-aligned; PAR_CHUNK_BYTES
    // is a multiple of every elem_size in use (16, 8, 4, 2, 1).
    let go_parallel = parallel && data.len() >= PAR_CHUNK_BYTES;

    if !go_parallel {
        return run_sequential(
            data, dtype, byte_order, allow_nan, allow_inf, elem_size, n_elements, 0,
        );
    }

    #[cfg(feature = "threads")]
    {
        run_parallel(
            data, dtype, byte_order, allow_nan, allow_inf, elem_size, n_elements,
        )
    }
    // `threads` feature disabled: fall back to sequential.
    #[cfg(not(feature = "threads"))]
    {
        run_sequential(
            data, dtype, byte_order, allow_nan, allow_inf, elem_size, n_elements, 0,
        )
    }
}

/// Sequential forward scan starting at element index `base_index`.
#[allow(clippy::too_many_arguments)]
fn run_sequential<'a>(
    data: &'a [u8],
    dtype: Dtype,
    byte_order: ByteOrder,
    allow_nan: bool,
    allow_inf: bool,
    elem_size: usize,
    n_elements: usize,
    base_index: usize,
) -> Result<(Cow<'a, [u8]>, MaskSet)> {
    // Phase 1: classify every element, collecting mask bits and
    // rejecting any disallowed non-finite kinds.
    let mut masks = MaskSet::empty(n_elements);
    let mut saw_any = false;

    // Dispatch to per-dtype scanners.  Each sets bits in `masks` and
    // returns an error on the first disallowed non-finite.
    match dtype {
        Dtype::Float32 => scan_f32(
            data,
            byte_order,
            base_index,
            allow_nan,
            allow_inf,
            &mut masks,
            &mut saw_any,
        )?,
        Dtype::Float64 => scan_f64(
            data,
            byte_order,
            base_index,
            allow_nan,
            allow_inf,
            &mut masks,
            &mut saw_any,
        )?,
        Dtype::Float16 => scan_f16(
            data,
            byte_order,
            base_index,
            allow_nan,
            allow_inf,
            &mut masks,
            &mut saw_any,
        )?,
        Dtype::Bfloat16 => scan_bf16(
            data,
            byte_order,
            base_index,
            allow_nan,
            allow_inf,
            &mut masks,
            &mut saw_any,
        )?,
        Dtype::Complex64 => scan_complex64(
            data,
            byte_order,
            base_index,
            allow_nan,
            allow_inf,
            &mut masks,
            &mut saw_any,
        )?,
        Dtype::Complex128 => scan_complex128(
            data,
            byte_order,
            base_index,
            allow_nan,
            allow_inf,
            &mut masks,
            &mut saw_any,
        )?,
        _ => unreachable!("non-float dtypes short-circuit before dispatch"),
    }

    if !saw_any {
        // Finite-only input: zero-copy passthrough, empty mask set.
        return Ok((Cow::Borrowed(data), MaskSet::empty(n_elements)));
    }

    // Phase 2: build the substituted buffer by zeroing out the
    // positions flagged in the mask set.  Non-flagged positions keep
    // their original bytes — this preserves finite-value bit patterns
    // exactly through the encode path.
    let mut out = data.to_vec();
    zero_at_mask_positions(&mut out, elem_size, &masks);

    Ok((Cow::Owned(out), masks))
}

#[cfg(feature = "threads")]
#[allow(clippy::too_many_arguments)]
fn run_parallel<'a>(
    data: &'a [u8],
    dtype: Dtype,
    byte_order: ByteOrder,
    allow_nan: bool,
    allow_inf: bool,
    elem_size: usize,
    n_elements: usize,
) -> Result<(Cow<'a, [u8]>, MaskSet)> {
    use rayon::prelude::*;

    let chunk_elements = PAR_CHUNK_BYTES / elem_size;

    // Per-chunk scans collect local MaskSets; errors short-circuit
    // the parallel iterator.
    let per_chunk: Vec<MaskSet> = data
        .par_chunks(PAR_CHUNK_BYTES)
        .enumerate()
        .map(|(chunk_idx, chunk)| -> Result<MaskSet> {
            let base = chunk_idx * chunk_elements;
            let (_, masks) = run_sequential(
                chunk,
                dtype,
                byte_order,
                allow_nan,
                allow_inf,
                elem_size,
                chunk.len() / elem_size,
                base,
            )?;
            Ok(masks)
        })
        .collect::<Result<Vec<_>>>()?;

    // Assemble global MaskSet from per-chunk results.
    let mut masks = MaskSet::empty(n_elements);
    let mut saw_any = false;
    for (chunk_idx, chunk_masks) in per_chunk.into_iter().enumerate() {
        if !chunk_masks.is_empty() {
            saw_any = true;
            let base = chunk_idx * chunk_elements;
            masks.merge_chunk(chunk_masks, base);
        }
    }

    if !saw_any {
        return Ok((Cow::Borrowed(data), MaskSet::empty(n_elements)));
    }

    let mut out = data.to_vec();
    zero_at_mask_positions(&mut out, elem_size, &masks);
    Ok((Cow::Owned(out), masks))
}

/// For every position set in any mask, overwrite `elem_size` bytes
/// with zero.  The canonical zero bit pattern works for all IEEE 754
/// float formats and their complex variants: +0.0 real, +0.0 imag.
fn zero_at_mask_positions(buf: &mut [u8], elem_size: usize, masks: &MaskSet) {
    let zero_position = |buf: &mut [u8], pos: usize| {
        let start = pos * elem_size;
        // Safe: mask length == n_elements == buf.len() / elem_size,
        // so start + elem_size <= buf.len().
        for b in &mut buf[start..start + elem_size] {
            *b = 0;
        }
    };

    if let Some(m) = &masks.nan {
        for (i, &b) in m.iter().enumerate() {
            if b {
                zero_position(buf, i);
            }
        }
    }
    if let Some(m) = &masks.pos_inf {
        for (i, &b) in m.iter().enumerate() {
            if b {
                zero_position(buf, i);
            }
        }
    }
    if let Some(m) = &masks.neg_inf {
        for (i, &b) in m.iter().enumerate() {
            if b {
                zero_position(buf, i);
            }
        }
    }
}

// ── Per-dtype classifiers ──────────────────────────────────────────────────
//
// Each classifier walks `chunk` one element at a time.  On a disallowed
// non-finite kind it returns an [`TensogramError::Encoding`] with the
// element index (offset by `base_index` for parallel chunks).  Allowed
// non-finite kinds set the corresponding mask bit and flip `saw_any`.

#[allow(clippy::too_many_arguments)]
fn scan_f32(
    chunk: &[u8],
    byte_order: ByteOrder,
    base_index: usize,
    allow_nan: bool,
    allow_inf: bool,
    masks: &mut MaskSet,
    saw_any: &mut bool,
) -> Result<()> {
    for (i, c) in chunk.chunks_exact(4).enumerate() {
        let &[b0, b1, b2, b3] = c else { continue };
        let val = match byte_order {
            ByteOrder::Big => f32::from_be_bytes([b0, b1, b2, b3]),
            ByteOrder::Little => f32::from_le_bytes([b0, b1, b2, b3]),
        };
        classify_scalar(val.is_nan(), val.is_infinite(), val.is_sign_positive()).dispatch(
            i,
            base_index + i,
            "float32",
            None,
            allow_nan,
            allow_inf,
            masks,
            saw_any,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn scan_f64(
    chunk: &[u8],
    byte_order: ByteOrder,
    base_index: usize,
    allow_nan: bool,
    allow_inf: bool,
    masks: &mut MaskSet,
    saw_any: &mut bool,
) -> Result<()> {
    for (i, c) in chunk.chunks_exact(8).enumerate() {
        let &[b0, b1, b2, b3, b4, b5, b6, b7] = c else {
            continue;
        };
        let val = match byte_order {
            ByteOrder::Big => f64::from_be_bytes([b0, b1, b2, b3, b4, b5, b6, b7]),
            ByteOrder::Little => f64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]),
        };
        classify_scalar(val.is_nan(), val.is_infinite(), val.is_sign_positive()).dispatch(
            i,
            base_index + i,
            "float64",
            None,
            allow_nan,
            allow_inf,
            masks,
            saw_any,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn scan_f16(
    chunk: &[u8],
    byte_order: ByteOrder,
    base_index: usize,
    allow_nan: bool,
    allow_inf: bool,
    masks: &mut MaskSet,
    saw_any: &mut bool,
) -> Result<()> {
    for (i, c) in chunk.chunks_exact(2).enumerate() {
        let &[b0, b1] = c else { continue };
        let bits = match byte_order {
            ByteOrder::Big => u16::from_be_bytes([b0, b1]),
            ByteOrder::Little => u16::from_le_bytes([b0, b1]),
        };
        let exp = (bits >> 10) & 0x1F;
        if exp != 0x1F {
            continue;
        }
        let mantissa = bits & 0x03FF;
        let positive = (bits & 0x8000) == 0;
        let kind = if mantissa != 0 {
            Classification::Nan
        } else if positive {
            Classification::PosInf
        } else {
            Classification::NegInf
        };
        kind.dispatch(
            i,
            base_index + i,
            "float16",
            None,
            allow_nan,
            allow_inf,
            masks,
            saw_any,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn scan_bf16(
    chunk: &[u8],
    byte_order: ByteOrder,
    base_index: usize,
    allow_nan: bool,
    allow_inf: bool,
    masks: &mut MaskSet,
    saw_any: &mut bool,
) -> Result<()> {
    for (i, c) in chunk.chunks_exact(2).enumerate() {
        let &[b0, b1] = c else { continue };
        let bits = match byte_order {
            ByteOrder::Big => u16::from_be_bytes([b0, b1]),
            ByteOrder::Little => u16::from_le_bytes([b0, b1]),
        };
        let exp = (bits >> 7) & 0xFF;
        if exp != 0xFF {
            continue;
        }
        let mantissa = bits & 0x7F;
        let positive = (bits & 0x8000) == 0;
        let kind = if mantissa != 0 {
            Classification::Nan
        } else if positive {
            Classification::PosInf
        } else {
            Classification::NegInf
        };
        kind.dispatch(
            i,
            base_index + i,
            "bfloat16",
            None,
            allow_nan,
            allow_inf,
            masks,
            saw_any,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn scan_complex64(
    chunk: &[u8],
    byte_order: ByteOrder,
    base_index: usize,
    allow_nan: bool,
    allow_inf: bool,
    masks: &mut MaskSet,
    saw_any: &mut bool,
) -> Result<()> {
    for (i, c) in chunk.chunks_exact(8).enumerate() {
        let &[r0, r1, r2, r3, im0, im1, im2, im3] = c else {
            continue;
        };
        let (real, imag) = match byte_order {
            ByteOrder::Big => (
                f32::from_be_bytes([r0, r1, r2, r3]),
                f32::from_be_bytes([im0, im1, im2, im3]),
            ),
            ByteOrder::Little => (
                f32::from_le_bytes([r0, r1, r2, r3]),
                f32::from_le_bytes([im0, im1, im2, im3]),
            ),
        };
        let (kind, component) = classify_complex(real, imag);
        kind.dispatch(
            i,
            base_index + i,
            "complex64",
            component,
            allow_nan,
            allow_inf,
            masks,
            saw_any,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn scan_complex128(
    chunk: &[u8],
    byte_order: ByteOrder,
    base_index: usize,
    allow_nan: bool,
    allow_inf: bool,
    masks: &mut MaskSet,
    saw_any: &mut bool,
) -> Result<()> {
    for (i, c) in chunk.chunks_exact(16).enumerate() {
        // Destructure the full 16-byte complex128 element so we can
        // parse real and imaginary components independently of byte
        // order.
        let &[
            r0,
            r1,
            r2,
            r3,
            r4,
            r5,
            r6,
            r7,
            im0,
            im1,
            im2,
            im3,
            im4,
            im5,
            im6,
            im7,
        ] = c
        else {
            continue;
        };
        let (real, imag) = match byte_order {
            ByteOrder::Big => (
                f64::from_be_bytes([r0, r1, r2, r3, r4, r5, r6, r7]),
                f64::from_be_bytes([im0, im1, im2, im3, im4, im5, im6, im7]),
            ),
            ByteOrder::Little => (
                f64::from_le_bytes([r0, r1, r2, r3, r4, r5, r6, r7]),
                f64::from_le_bytes([im0, im1, im2, im3, im4, im5, im6, im7]),
            ),
        };
        let (kind, component) = classify_complex_f64(real, imag);
        kind.dispatch(
            i,
            base_index + i,
            "complex128",
            component,
            allow_nan,
            allow_inf,
            masks,
            saw_any,
        )?;
    }
    Ok(())
}

// ── Classification ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Classification {
    Finite,
    Nan,
    PosInf,
    NegInf,
}

impl Classification {
    /// Apply the `allow_*` policy to this classification.
    ///
    /// `local_index` is the position within the current `MaskSet`
    /// (always 0-based for the call; the parallel merge step
    /// re-offsets).  `global_index` is the index reported in error
    /// messages so parallel workers produce user-facing indices that
    /// match sequential output.  The two collapse to the same value
    /// in the sequential path.
    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        self,
        local_index: usize,
        global_index: usize,
        dtype_name: &'static str,
        component: Option<&'static str>,
        allow_nan: bool,
        allow_inf: bool,
        masks: &mut MaskSet,
        saw_any: &mut bool,
    ) -> Result<()> {
        match self {
            Classification::Finite => Ok(()),
            Classification::Nan => {
                if !allow_nan {
                    return Err(nan_err(global_index, dtype_name, component));
                }
                masks.set_nan(local_index);
                *saw_any = true;
                Ok(())
            }
            Classification::PosInf => {
                if !allow_inf {
                    return Err(inf_err(global_index, dtype_name, component, true));
                }
                masks.set_pos_inf(local_index);
                *saw_any = true;
                Ok(())
            }
            Classification::NegInf => {
                if !allow_inf {
                    return Err(inf_err(global_index, dtype_name, component, false));
                }
                masks.set_neg_inf(local_index);
                *saw_any = true;
                Ok(())
            }
        }
    }
}

fn classify_scalar(is_nan: bool, is_infinite: bool, is_sign_positive: bool) -> Classification {
    if is_nan {
        Classification::Nan
    } else if is_infinite {
        if is_sign_positive {
            Classification::PosInf
        } else {
            Classification::NegInf
        }
    } else {
        Classification::Finite
    }
}

/// Complex priority: NaN > +Inf > -Inf.  Returns the kind to emit AND
/// the component (`"real"` / `"imaginary"`) that first triggered it
/// — used in error messages when the kind is disallowed.
fn classify_complex(real: f32, imag: f32) -> (Classification, Option<&'static str>) {
    if real.is_nan() {
        return (Classification::Nan, Some("real"));
    }
    if imag.is_nan() {
        return (Classification::Nan, Some("imaginary"));
    }
    if real.is_infinite() && real.is_sign_positive() {
        return (Classification::PosInf, Some("real"));
    }
    if imag.is_infinite() && imag.is_sign_positive() {
        return (Classification::PosInf, Some("imaginary"));
    }
    if real.is_infinite() {
        return (Classification::NegInf, Some("real"));
    }
    if imag.is_infinite() {
        return (Classification::NegInf, Some("imaginary"));
    }
    (Classification::Finite, None)
}

fn classify_complex_f64(real: f64, imag: f64) -> (Classification, Option<&'static str>) {
    if real.is_nan() {
        return (Classification::Nan, Some("real"));
    }
    if imag.is_nan() {
        return (Classification::Nan, Some("imaginary"));
    }
    if real.is_infinite() && real.is_sign_positive() {
        return (Classification::PosInf, Some("real"));
    }
    if imag.is_infinite() && imag.is_sign_positive() {
        return (Classification::PosInf, Some("imaginary"));
    }
    if real.is_infinite() {
        return (Classification::NegInf, Some("real"));
    }
    if imag.is_infinite() {
        return (Classification::NegInf, Some("imaginary"));
    }
    (Classification::Finite, None)
}

// ── Error message builders — match finite_check's format ───────────────────

fn nan_err(index: usize, dtype: &str, component: Option<&str>) -> TensogramError {
    let suffix = component
        .map(|c| format!(" ({c} component)"))
        .unwrap_or_default();
    TensogramError::Encoding(format!(
        "strict-NaN check: NaN at element {index} of {dtype} array{suffix}; \
         pass allow_nan=true to substitute with 0.0 and record positions in a mask"
    ))
}

fn inf_err(index: usize, dtype: &str, component: Option<&str>, positive: bool) -> TensogramError {
    let sign = if positive { "+Inf" } else { "-Inf" };
    let suffix = component
        .map(|c| format!(" ({c} component)"))
        .unwrap_or_default();
    TensogramError::Encoding(format!(
        "strict-Inf check: {sign} at element {index} of {dtype} array{suffix}; \
         pass allow_inf=true to substitute with 0.0 and record positions in a mask"
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
    fn non_float_dtype_zero_cost_borrowed() {
        let data = vec![0xFFu8; 64];
        let (out, masks) =
            substitute_and_mask(&data, Dtype::Uint32, ByteOrder::native(), true, true, false)
                .unwrap();
        assert!(matches!(out, Cow::Borrowed(_)));
        assert!(masks.is_empty());
    }

    #[test]
    fn finite_float_input_zero_cost_borrowed() {
        let data = f64_bytes(&[1.0, 2.0, 3.0, 4.0, -1.0]);
        let (out, masks) = substitute_and_mask(
            &data,
            Dtype::Float64,
            ByteOrder::native(),
            true,
            true,
            false,
        )
        .unwrap();
        assert!(matches!(out, Cow::Borrowed(_)));
        assert!(masks.is_empty());
    }

    #[test]
    fn disallowed_nan_errors_with_hint() {
        let data = f32_bytes(&[1.0, f32::NAN]);
        let err = substitute_and_mask(
            &data,
            Dtype::Float32,
            ByteOrder::native(),
            false,
            false,
            false,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("element 1"), "got: {msg}");
        assert!(msg.contains("allow_nan=true"), "got: {msg}");
    }

    #[test]
    fn disallowed_pos_inf_errors_with_hint() {
        let data = f64_bytes(&[1.0, f64::INFINITY]);
        let err = substitute_and_mask(
            &data,
            Dtype::Float64,
            ByteOrder::native(),
            false,
            false,
            false,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("+Inf"), "got: {msg}");
        assert!(msg.contains("allow_inf=true"), "got: {msg}");
    }

    #[test]
    fn allow_nan_produces_mask_and_substitutes_with_zero() {
        let data = f64_bytes(&[1.0, f64::NAN, 3.0, f64::NAN]);
        let (out, masks) = substitute_and_mask(
            &data,
            Dtype::Float64,
            ByteOrder::native(),
            true,
            false,
            false,
        )
        .unwrap();
        // Byte comparison: element 1 and 3 zeroed out, others untouched.
        let got = out.as_ref();
        let expected = f64_bytes(&[1.0, 0.0, 3.0, 0.0]);
        assert_eq!(got, expected.as_slice());
        // NaN mask populated at positions 1 and 3.
        let nan = masks.nan.unwrap();
        assert_eq!(nan, vec![false, true, false, true]);
        assert!(masks.pos_inf.is_none());
        assert!(masks.neg_inf.is_none());
    }

    #[test]
    fn allow_inf_masks_both_signs_separately() {
        let data = f64_bytes(&[0.0, f64::INFINITY, f64::NEG_INFINITY, 2.0]);
        let (_, masks) = substitute_and_mask(
            &data,
            Dtype::Float64,
            ByteOrder::native(),
            false,
            true,
            false,
        )
        .unwrap();
        assert_eq!(masks.pos_inf.unwrap(), vec![false, true, false, false]);
        assert_eq!(masks.neg_inf.unwrap(), vec![false, false, true, false]);
        assert!(masks.nan.is_none());
    }

    #[test]
    fn complex64_nan_wins_over_inf_with_priority_rule() {
        // (NaN + Inf*i): NaN dominates — element contributes to nan mask only.
        let data: Vec<u8> = [f32::NAN, f32::INFINITY]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let (_, masks) = substitute_and_mask(
            &data,
            Dtype::Complex64,
            ByteOrder::native(),
            true,
            true,
            false,
        )
        .unwrap();
        assert_eq!(masks.nan.unwrap(), vec![true]);
        assert!(masks.pos_inf.is_none());
        assert!(masks.neg_inf.is_none());
    }

    #[test]
    fn complex64_pos_inf_wins_over_neg_inf() {
        // (-Inf + +Inf*i): +Inf dominates — element contributes to pos_inf mask only.
        let data: Vec<u8> = [f32::NEG_INFINITY, f32::INFINITY]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let (_, masks) = substitute_and_mask(
            &data,
            Dtype::Complex64,
            ByteOrder::native(),
            true,
            true,
            false,
        )
        .unwrap();
        assert_eq!(masks.pos_inf.unwrap(), vec![true]);
        assert!(masks.nan.is_none());
        assert!(masks.neg_inf.is_none());
    }

    #[test]
    fn complex64_substitution_zeroes_both_components() {
        let data: Vec<u8> = [1.0_f32, f32::NAN]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let (out, _) = substitute_and_mask(
            &data,
            Dtype::Complex64,
            ByteOrder::native(),
            true,
            false,
            false,
        )
        .unwrap();
        // Both real and imaginary components must be 0.0 after substitution.
        let got = out.as_ref();
        let expected: Vec<u8> = [0.0_f32, 0.0_f32]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        assert_eq!(got, expected.as_slice());
    }

    #[test]
    fn float16_nan_detected_and_substituted() {
        // 0x7E00 = NaN (exp=0x1F, mantissa != 0).
        let bits: u16 = 0x7E00;
        let data = bits.to_ne_bytes().to_vec();
        let (out, masks) = substitute_and_mask(
            &data,
            Dtype::Float16,
            ByteOrder::native(),
            true,
            false,
            false,
        )
        .unwrap();
        // Substituted: 0x0000 (both bytes zero).
        assert_eq!(out.as_ref(), &[0u8, 0u8][..]);
        assert_eq!(masks.nan.unwrap(), vec![true]);
    }

    #[test]
    fn bfloat16_neg_inf_detected_with_sign() {
        // 0xFF80 = -Inf for bfloat16 (exp=0xFF, mantissa=0, sign=1).
        let bits: u16 = 0xFF80;
        let data = bits.to_ne_bytes().to_vec();
        let (_, masks) = substitute_and_mask(
            &data,
            Dtype::Bfloat16,
            ByteOrder::native(),
            false,
            true,
            false,
        )
        .unwrap();
        assert_eq!(masks.neg_inf.unwrap(), vec![true]);
        assert!(masks.pos_inf.is_none());
    }

    #[test]
    fn mixed_nan_and_inf_build_separate_masks() {
        let data = f64_bytes(&[
            1.0,
            f64::NAN,
            2.0,
            f64::INFINITY,
            3.0,
            f64::NEG_INFINITY,
            4.0,
            f64::NAN,
        ]);
        let (out, masks) = substitute_and_mask(
            &data,
            Dtype::Float64,
            ByteOrder::native(),
            true,
            true,
            false,
        )
        .unwrap();
        let expected = f64_bytes(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]);
        assert_eq!(out.as_ref(), expected.as_slice());
        assert_eq!(
            masks.nan.unwrap(),
            vec![false, true, false, false, false, false, false, true]
        );
        assert_eq!(
            masks.pos_inf.unwrap(),
            vec![false, false, false, true, false, false, false, false]
        );
        assert_eq!(
            masks.neg_inf.unwrap(),
            vec![false, false, false, false, false, true, false, false]
        );
    }

    #[cfg(feature = "threads")]
    #[test]
    fn parallel_path_matches_sequential() {
        let mut values: Vec<f64> = (0..16_384).map(|i| i as f64).collect();
        values[9_001] = f64::NAN;
        values[9_002] = f64::INFINITY;
        values[16_000] = f64::NEG_INFINITY;
        let data = f64_bytes(&values);

        let (seq_out, seq_masks) = substitute_and_mask(
            &data,
            Dtype::Float64,
            ByteOrder::native(),
            true,
            true,
            false,
        )
        .unwrap();
        let (par_out, par_masks) =
            substitute_and_mask(&data, Dtype::Float64, ByteOrder::native(), true, true, true)
                .unwrap();

        assert_eq!(seq_out.as_ref(), par_out.as_ref());
        assert_eq!(seq_masks.nan, par_masks.nan);
        assert_eq!(seq_masks.pos_inf, par_masks.pos_inf);
        assert_eq!(seq_masks.neg_inf, par_masks.neg_inf);
    }

    #[test]
    fn empty_input_passes_through() {
        let data: Vec<u8> = vec![];
        let (out, masks) = substitute_and_mask(
            &data,
            Dtype::Float64,
            ByteOrder::native(),
            true,
            true,
            false,
        )
        .unwrap();
        assert!(matches!(out, Cow::Borrowed(_)));
        assert!(masks.is_empty());
    }
}
