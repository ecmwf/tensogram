// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Level 4: Fidelity validation — full decode, size check, mask-aware
//! NaN / Inf detection.
//!
//! When a data-object frame carries a `masks` sub-map
//! (`NTensorFrame`, see `plans/BITMASK_FRAME.md` §3.2), the
//! validator decompresses the per-kind bitmasks and cross-checks
//! the reconstructed decoded output.  At every mask bit set to `1`,
//! the matching canonical non-finite value is **expected**.  At
//! every other position, any NaN / ±Inf indicates a corruption —
//! either in the payload pipeline output or in the mask itself.
//!
//! Without masks, the scanner keeps its pre-0.17 behaviour: any
//! NaN / Inf in the decoded payload is an error.

use crate::dtype::Dtype;
use crate::encode::build_pipeline_config;
use crate::restore::DecodedMaskSet;
use tensogram_encodings::ByteOrder;

use super::types::*;

/// Non-finite kinds the fidelity scanner reports.  Mirrors
/// [`crate::substitute_and_mask::Classification`] but scoped to
/// `(issue_code, optional_component_name)` — the shape the issue
/// builder needs.
#[derive(Debug, Clone, Copy)]
enum NonFinite {
    Nan,
    Inf,
}

impl NonFinite {
    fn issue_code(self) -> IssueCode {
        match self {
            NonFinite::Nan => IssueCode::NanDetected,
            NonFinite::Inf => IssueCode::InfDetected,
        }
    }

    fn label(self) -> &'static str {
        match self {
            NonFinite::Nan => "NaN",
            NonFinite::Inf => "Inf",
        }
    }
}

/// Per-element outcome: a non-finite kind (plus optional component
/// name for complex dtypes), or `None` when the element is finite.
type ElementReport = Option<(NonFinite, Option<&'static str>)>;

/// Run fidelity checks on all objects.
///
/// For each object:
/// 1. Ensure decoded bytes exist (reuse Level 3 cache or decode raw payload).
/// 2. Verify decoded size matches shape * dtype byte width.
/// 3. Scan float types for NaN/Inf values (errors, not warnings).
pub(crate) fn validate_fidelity(
    objects: &mut [ObjectContext<'_>],
    issues: &mut Vec<ValidationIssue>,
) {
    for (i, obj) in objects.iter_mut().enumerate() {
        let desc = match obj.descriptor.as_ref() {
            Some(d) => d,
            None => continue, // Already reported at Level 2
        };

        // Get decoded bytes: reuse Level 3 cache, scan raw payload in-place, or decode now
        let decoded: &[u8] = match &obj.decode_state {
            DecodeState::Decoded(bytes) => bytes,
            DecodeState::DecodeFailed => continue, // Already reported at Level 3
            DecodeState::NotDecoded => {
                if desc.encoding == "none" && desc.filter == "none" && desc.compression == "none" {
                    // Raw object: payload is the decoded bytes
                    obj.payload
                } else {
                    // Non-raw object — Level 3 didn't run, decode now.
                    // Currently unreachable via validate_message (Fidelity implies
                    // Integrity which decodes all non-raw objects), but kept as a
                    // safety net for direct API callers.
                    let shape_product = desc
                        .shape
                        .iter()
                        .try_fold(1u64, |acc, &x| acc.checked_mul(x));
                    let num_elements = match shape_product.and_then(|p| usize::try_from(p).ok()) {
                        Some(n) => n,
                        None => {
                            issues.push(err(
                                IssueCode::DecodeObjectFailed,
                                ValidationLevel::Fidelity,
                                Some(i),
                                Some(obj.frame_offset),
                                "cannot compute element count from shape".to_string(),
                            ));
                            continue;
                        }
                    };
                    match build_pipeline_config(desc, num_elements, desc.dtype) {
                        Ok(config) => {
                            match tensogram_encodings::pipeline::decode_pipeline(
                                obj.payload,
                                &config,
                                false,
                            ) {
                                Ok(decoded_bytes) => {
                                    obj.decode_state = DecodeState::Decoded(decoded_bytes);
                                    // Safe: just set to Decoded above
                                    let DecodeState::Decoded(b) = &obj.decode_state else {
                                        continue;
                                    };
                                    b
                                }
                                Err(e) => {
                                    obj.decode_state = DecodeState::DecodeFailed;
                                    issues.push(err(
                                        IssueCode::DecodeObjectFailed,
                                        ValidationLevel::Fidelity,
                                        Some(i),
                                        Some(obj.frame_offset),
                                        format!("full decode failed: {e}"),
                                    ));
                                    continue;
                                }
                            }
                        }
                        Err(e) => {
                            obj.decode_state = DecodeState::DecodeFailed;
                            issues.push(err(
                                IssueCode::DecodeObjectFailed,
                                ValidationLevel::Fidelity,
                                Some(i),
                                Some(obj.frame_offset),
                                format!("cannot build pipeline config: {e}"),
                            ));
                            continue;
                        }
                    }
                }
            }
        };

        // Decoded-size check (unconditional, including size 0)
        let shape_product = desc
            .shape
            .iter()
            .try_fold(1u64, |acc, &x| acc.checked_mul(x));
        let expected_size = match shape_product {
            Some(product) => {
                if desc.dtype == Dtype::Bitmask {
                    usize::try_from(product.div_ceil(8)).ok()
                } else {
                    usize::try_from(product)
                        .ok()
                        .and_then(|p| p.checked_mul(desc.dtype.byte_width()))
                }
            }
            None => {
                issues.push(err(
                    IssueCode::DecodedSizeMismatch,
                    ValidationLevel::Fidelity,
                    Some(i),
                    Some(obj.frame_offset),
                    "shape product overflows, cannot verify decoded size".to_string(),
                ));
                continue;
            }
        };
        let expected_size = match expected_size {
            Some(s) => s,
            None => {
                issues.push(err(
                    IssueCode::DecodedSizeMismatch,
                    ValidationLevel::Fidelity,
                    Some(i),
                    Some(obj.frame_offset),
                    "expected decoded size overflows usize".to_string(),
                ));
                continue;
            }
        };

        if decoded.len() != expected_size {
            issues.push(err(
                IssueCode::DecodedSizeMismatch,
                ValidationLevel::Fidelity,
                Some(i),
                Some(obj.frame_offset),
                format!(
                    "decoded size {} != expected {} (shape {:?}, dtype {})",
                    decoded.len(),
                    expected_size,
                    desc.shape,
                    desc.dtype
                ),
            ));
            continue;
        }

        // Decompress the mask set (if any) so the scanner knows which
        // non-finite positions to expect.
        let mask_set = match crate::restore::decode_mask_set(desc, obj.mask_region) {
            Ok(ms) => ms,
            Err(e) => {
                issues.push(err(
                    IssueCode::DecodeObjectFailed,
                    ValidationLevel::Fidelity,
                    Some(i),
                    Some(obj.frame_offset),
                    format!("bitmask decode failed: {e}"),
                ));
                continue;
            }
        };

        // NaN/Inf scan for float types — each scanner consults the
        // mask set and only reports NaN/Inf at positions the mask did
        // not claim.
        let byte_order = desc.byte_order;
        match desc.dtype {
            Dtype::Float32 => scan_f32(decoded, byte_order, i, obj.frame_offset, &mask_set, issues),
            Dtype::Float64 => scan_f64(decoded, byte_order, i, obj.frame_offset, &mask_set, issues),
            Dtype::Float16 => scan_f16(decoded, byte_order, i, obj.frame_offset, &mask_set, issues),
            Dtype::Bfloat16 => {
                scan_bf16(decoded, byte_order, i, obj.frame_offset, &mask_set, issues)
            }
            Dtype::Complex64 => {
                scan_complex64(decoded, byte_order, i, obj.frame_offset, &mask_set, issues)
            }
            Dtype::Complex128 => {
                scan_complex128(decoded, byte_order, i, obj.frame_offset, &mask_set, issues)
            }
            _ => {} // Integer/bitmask types: no NaN/Inf possible
        }
    }
}

// ── Mask-set helpers ────────────────────────────────────────────────────────

/// Return whether `idx` is flagged in any of the three masks — i.e.
/// the encoder said the element was non-finite.  The scanner uses
/// this to suppress NaN/Inf reports at expected positions.
fn index_is_masked(mask_set: &DecodedMaskSet, idx: usize) -> bool {
    let check = |bits: Option<&Vec<bool>>| bits.is_some_and(|b| b.get(idx).copied() == Some(true));
    check(mask_set.nan.as_ref())
        || check(mask_set.pos_inf.as_ref())
        || check(mask_set.neg_inf.as_ref())
}

// ── Float scanners ──────────────────────────────────────────────────────────
//
// Each scanner is a thin wrapper around [`scan_floats`] that binds
// the dtype-specific constants (name, element size) and supplies a
// per-element classifier closure.  The scaffolding — size check,
// chunk iteration, mask-aware issue emission — lives in one place.

fn scan_f32(
    data: &[u8],
    byte_order: ByteOrder,
    obj_idx: usize,
    frame_offset: usize,
    mask_set: &DecodedMaskSet,
    issues: &mut Vec<ValidationIssue>,
) {
    scan_floats(
        data,
        4,
        "float32",
        obj_idx,
        frame_offset,
        mask_set,
        issues,
        |chunk| classify_scalar_f32(chunk, byte_order),
    );
}

fn scan_f64(
    data: &[u8],
    byte_order: ByteOrder,
    obj_idx: usize,
    frame_offset: usize,
    mask_set: &DecodedMaskSet,
    issues: &mut Vec<ValidationIssue>,
) {
    scan_floats(
        data,
        8,
        "float64",
        obj_idx,
        frame_offset,
        mask_set,
        issues,
        |chunk| classify_scalar_f64(chunk, byte_order),
    );
}

fn scan_f16(
    data: &[u8],
    byte_order: ByteOrder,
    obj_idx: usize,
    frame_offset: usize,
    mask_set: &DecodedMaskSet,
    issues: &mut Vec<ValidationIssue>,
) {
    scan_floats(
        data,
        2,
        "float16",
        obj_idx,
        frame_offset,
        mask_set,
        issues,
        |chunk| classify_half(chunk, byte_order, HalfLayout::F16),
    );
}

fn scan_bf16(
    data: &[u8],
    byte_order: ByteOrder,
    obj_idx: usize,
    frame_offset: usize,
    mask_set: &DecodedMaskSet,
    issues: &mut Vec<ValidationIssue>,
) {
    scan_floats(
        data,
        2,
        "bfloat16",
        obj_idx,
        frame_offset,
        mask_set,
        issues,
        |chunk| classify_half(chunk, byte_order, HalfLayout::BF16),
    );
}

fn scan_complex64(
    data: &[u8],
    byte_order: ByteOrder,
    obj_idx: usize,
    frame_offset: usize,
    mask_set: &DecodedMaskSet,
    issues: &mut Vec<ValidationIssue>,
) {
    scan_floats(
        data,
        8,
        "complex64",
        obj_idx,
        frame_offset,
        mask_set,
        issues,
        |chunk| classify_complex64(chunk, byte_order),
    );
}

fn scan_complex128(
    data: &[u8],
    byte_order: ByteOrder,
    obj_idx: usize,
    frame_offset: usize,
    mask_set: &DecodedMaskSet,
    issues: &mut Vec<ValidationIssue>,
) {
    scan_floats(
        data,
        16,
        "complex128",
        obj_idx,
        frame_offset,
        mask_set,
        issues,
        |chunk| classify_complex128(chunk, byte_order),
    );
}

/// Shared scaffolding for every float / complex scanner: size check,
/// chunk iteration, mask-aware issue emission.  Callers supply:
/// - `elem_size`: byte width of one element (2, 4, 8, or 16).
/// - `dtype_name`: dtype string for the size-mismatch error.
/// - `classify`: closure that inspects a single `elem_size`-wide
///   chunk and returns an [`ElementReport`] (None = finite, Some =
///   non-finite kind + optional component name for complex types).
///
/// Reports at most one issue per call — matches the pre-Pass-4
/// semantics ("first occurrence only").
#[allow(clippy::too_many_arguments)]
fn scan_floats<F>(
    data: &[u8],
    elem_size: usize,
    dtype_name: &'static str,
    obj_idx: usize,
    frame_offset: usize,
    mask_set: &DecodedMaskSet,
    issues: &mut Vec<ValidationIssue>,
    classify: F,
) where
    F: Fn(&[u8]) -> ElementReport,
{
    let chunks = data.chunks_exact(elem_size);
    if !chunks.remainder().is_empty() {
        issues.push(err(
            IssueCode::DecodedSizeMismatch,
            ValidationLevel::Fidelity,
            Some(obj_idx),
            Some(frame_offset),
            format!(
                "decoded size {} is not a multiple of {elem_size} ({dtype_name})",
                data.len()
            ),
        ));
        return;
    }
    for (idx, chunk) in chunks.enumerate() {
        let Some((kind, component)) = classify(chunk) else {
            continue;
        };
        if index_is_masked(mask_set, idx) {
            continue;
        }
        let component_suffix = component
            .map(|c| format!(" ({c} component, not recorded in mask)"))
            .unwrap_or_else(|| " (not recorded in mask)".to_string());
        issues.push(err(
            kind.issue_code(),
            ValidationLevel::Fidelity,
            Some(obj_idx),
            Some(frame_offset),
            format!(
                "unexpected {} at element {idx}{component_suffix}",
                kind.label()
            ),
        ));
        return; // Report first occurrence only.
    }
}

// ── Per-dtype classifiers ───────────────────────────────────────────────────
//
// Each classifier takes a chunk of bytes known to be exactly one
// element wide (the caller's chunks_exact guarantees this) and
// returns an ElementReport.  The `chunks_exact` slice always has
// the expected length, so the slice-pattern `else { return None }`
// branches are effectively unreachable — we use a defensive None
// return to avoid panicking if the std contract ever changes.

fn classify_scalar_f32(chunk: &[u8], byte_order: ByteOrder) -> ElementReport {
    let &[b0, b1, b2, b3] = chunk else {
        return None;
    };
    let bytes = [b0, b1, b2, b3];
    let val = match byte_order {
        ByteOrder::Big => f32::from_be_bytes(bytes),
        ByteOrder::Little => f32::from_le_bytes(bytes),
    };
    classify_float_bits(val.is_nan(), val.is_infinite()).map(|k| (k, None))
}

fn classify_scalar_f64(chunk: &[u8], byte_order: ByteOrder) -> ElementReport {
    let &[b0, b1, b2, b3, b4, b5, b6, b7] = chunk else {
        return None;
    };
    let bytes = [b0, b1, b2, b3, b4, b5, b6, b7];
    let val = match byte_order {
        ByteOrder::Big => f64::from_be_bytes(bytes),
        ByteOrder::Little => f64::from_le_bytes(bytes),
    };
    classify_float_bits(val.is_nan(), val.is_infinite()).map(|k| (k, None))
}

/// IEEE half layout — exponent bit positions / widths.
#[derive(Clone, Copy)]
enum HalfLayout {
    /// IEEE 754 half: sign(1) + exp(5) + mantissa(10).
    F16,
    /// BFloat16: sign(1) + exp(8) + mantissa(7).
    BF16,
}

impl HalfLayout {
    fn exp_all_ones(self) -> u16 {
        match self {
            HalfLayout::F16 => 0x1F,
            HalfLayout::BF16 => 0xFF,
        }
    }
    fn exp_shift(self) -> u32 {
        match self {
            HalfLayout::F16 => 10,
            HalfLayout::BF16 => 7,
        }
    }
    fn mantissa_mask(self) -> u16 {
        match self {
            HalfLayout::F16 => 0x03FF,
            HalfLayout::BF16 => 0x7F,
        }
    }
}

fn classify_half(chunk: &[u8], byte_order: ByteOrder, layout: HalfLayout) -> ElementReport {
    let &[b0, b1] = chunk else { return None };
    let bits = match byte_order {
        ByteOrder::Big => u16::from_be_bytes([b0, b1]),
        ByteOrder::Little => u16::from_le_bytes([b0, b1]),
    };
    let exp = (bits >> layout.exp_shift()) & layout.exp_all_ones();
    if exp != layout.exp_all_ones() {
        return None;
    }
    let mantissa = bits & layout.mantissa_mask();
    let kind = if mantissa != 0 {
        NonFinite::Nan
    } else {
        NonFinite::Inf
    };
    Some((kind, None))
}

fn classify_complex64(chunk: &[u8], byte_order: ByteOrder) -> ElementReport {
    let &[r0, r1, r2, r3, i0, i1, i2, i3] = chunk else {
        return None;
    };
    let (real, imag) = match byte_order {
        ByteOrder::Big => (
            f32::from_be_bytes([r0, r1, r2, r3]),
            f32::from_be_bytes([i0, i1, i2, i3]),
        ),
        ByteOrder::Little => (
            f32::from_le_bytes([r0, r1, r2, r3]),
            f32::from_le_bytes([i0, i1, i2, i3]),
        ),
    };
    classify_complex_components(
        real.is_nan(),
        real.is_infinite(),
        imag.is_nan(),
        imag.is_infinite(),
    )
}

fn classify_complex128(chunk: &[u8], byte_order: ByteOrder) -> ElementReport {
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
        return None;
    };
    let (real, imag) = match byte_order {
        ByteOrder::Big => (
            f64::from_be_bytes([r0, r1, r2, r3, r4, r5, r6, r7]),
            f64::from_be_bytes([i0, i1, i2, i3, i4, i5, i6, i7]),
        ),
        ByteOrder::Little => (
            f64::from_le_bytes([r0, r1, r2, r3, r4, r5, r6, r7]),
            f64::from_le_bytes([i0, i1, i2, i3, i4, i5, i6, i7]),
        ),
    };
    classify_complex_components(
        real.is_nan(),
        real.is_infinite(),
        imag.is_nan(),
        imag.is_infinite(),
    )
}

/// Collapse (is_nan, is_infinite) into an [`ElementReport`] without
/// a component name.  Used by the scalar-float classifiers.
fn classify_float_bits(is_nan: bool, is_infinite: bool) -> Option<NonFinite> {
    if is_nan {
        Some(NonFinite::Nan)
    } else if is_infinite {
        Some(NonFinite::Inf)
    } else {
        None
    }
}

/// Report the first non-finite component of a complex value,
/// preserving the `"real"` / `"imaginary"` distinction used in
/// issue messages.  Mirrors the pre-Pass-4 scan behaviour.
fn classify_complex_components(
    real_nan: bool,
    real_inf: bool,
    imag_nan: bool,
    imag_inf: bool,
) -> ElementReport {
    if real_nan {
        return Some((NonFinite::Nan, Some("real")));
    }
    if real_inf {
        return Some((NonFinite::Inf, Some("real")));
    }
    if imag_nan {
        return Some((NonFinite::Nan, Some("imaginary")));
    }
    if imag_inf {
        return Some((NonFinite::Inf, Some("imaginary")));
    }
    None
}
