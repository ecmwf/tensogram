//! Level 4: Fidelity validation — full decode, size check, NaN/Inf detection.

use crate::dtype::Dtype;
use crate::encode::build_pipeline_config;
use tensogram_encodings::ByteOrder;

use super::types::*;

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
                    // Non-raw object — Level 3 didn't run, decode now
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

        // NaN/Inf scan for float types
        let byte_order = desc.byte_order;
        match desc.dtype {
            Dtype::Float32 => scan_f32(decoded, byte_order, i, obj.frame_offset, issues),
            Dtype::Float64 => scan_f64(decoded, byte_order, i, obj.frame_offset, issues),
            Dtype::Float16 => scan_f16(decoded, byte_order, i, obj.frame_offset, issues),
            Dtype::Bfloat16 => scan_bf16(decoded, byte_order, i, obj.frame_offset, issues),
            Dtype::Complex64 => scan_complex64(decoded, byte_order, i, obj.frame_offset, issues),
            Dtype::Complex128 => scan_complex128(decoded, byte_order, i, obj.frame_offset, issues),
            _ => {} // Integer/bitmask types: no NaN/Inf possible
        }
    }
}

// ── Float scanners ──────────────────────────────────────────────────────────

fn scan_f32(
    data: &[u8],
    byte_order: ByteOrder,
    obj_idx: usize,
    frame_offset: usize,
    issues: &mut Vec<ValidationIssue>,
) {
    let chunks = data.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        issues.push(err(
            IssueCode::DecodedSizeMismatch,
            ValidationLevel::Fidelity,
            Some(obj_idx),
            Some(frame_offset),
            format!(
                "decoded size {} is not a multiple of 4 (float32)",
                data.len()
            ),
        ));
        return;
    }
    for (idx, chunk) in chunks.enumerate() {
        let bytes: [u8; 4] = chunk.try_into().unwrap();
        let val = match byte_order {
            ByteOrder::Big => f32::from_be_bytes(bytes),
            ByteOrder::Little => f32::from_le_bytes(bytes),
        };
        if val.is_nan() {
            issues.push(err(
                IssueCode::NanDetected,
                ValidationLevel::Fidelity,
                Some(obj_idx),
                Some(frame_offset),
                format!("NaN at element {idx}"),
            ));
            return; // Report first occurrence only
        }
        if val.is_infinite() {
            issues.push(err(
                IssueCode::InfDetected,
                ValidationLevel::Fidelity,
                Some(obj_idx),
                Some(frame_offset),
                format!("Inf at element {idx}"),
            ));
            return;
        }
    }
}

fn scan_f64(
    data: &[u8],
    byte_order: ByteOrder,
    obj_idx: usize,
    frame_offset: usize,
    issues: &mut Vec<ValidationIssue>,
) {
    let chunks = data.chunks_exact(8);
    if !chunks.remainder().is_empty() {
        issues.push(err(
            IssueCode::DecodedSizeMismatch,
            ValidationLevel::Fidelity,
            Some(obj_idx),
            Some(frame_offset),
            format!(
                "decoded size {} is not a multiple of 8 (float64)",
                data.len()
            ),
        ));
        return;
    }
    for (idx, chunk) in chunks.enumerate() {
        let bytes: [u8; 8] = chunk.try_into().unwrap();
        let val = match byte_order {
            ByteOrder::Big => f64::from_be_bytes(bytes),
            ByteOrder::Little => f64::from_le_bytes(bytes),
        };
        if val.is_nan() {
            issues.push(err(
                IssueCode::NanDetected,
                ValidationLevel::Fidelity,
                Some(obj_idx),
                Some(frame_offset),
                format!("NaN at element {idx}"),
            ));
            return;
        }
        if val.is_infinite() {
            issues.push(err(
                IssueCode::InfDetected,
                ValidationLevel::Fidelity,
                Some(obj_idx),
                Some(frame_offset),
                format!("Inf at element {idx}"),
            ));
            return;
        }
    }
}

fn scan_f16(
    data: &[u8],
    byte_order: ByteOrder,
    obj_idx: usize,
    frame_offset: usize,
    issues: &mut Vec<ValidationIssue>,
) {
    let chunks = data.chunks_exact(2);
    if !chunks.remainder().is_empty() {
        issues.push(err(
            IssueCode::DecodedSizeMismatch,
            ValidationLevel::Fidelity,
            Some(obj_idx),
            Some(frame_offset),
            format!(
                "decoded size {} is not a multiple of 2 (float16)",
                data.len()
            ),
        ));
        return;
    }
    for (idx, chunk) in chunks.enumerate() {
        let bytes: [u8; 2] = chunk.try_into().unwrap();
        let bits = match byte_order {
            ByteOrder::Big => u16::from_be_bytes(bytes),
            ByteOrder::Little => u16::from_le_bytes(bytes),
        };
        // IEEE 754 half: sign(1) + exponent(5) + mantissa(10)
        let exp = (bits >> 10) & 0x1F;
        let mantissa = bits & 0x03FF;
        if exp == 0x1F {
            if mantissa != 0 {
                issues.push(err(
                    IssueCode::NanDetected,
                    ValidationLevel::Fidelity,
                    Some(obj_idx),
                    Some(frame_offset),
                    format!("NaN at element {idx}"),
                ));
                return;
            } else {
                issues.push(err(
                    IssueCode::InfDetected,
                    ValidationLevel::Fidelity,
                    Some(obj_idx),
                    Some(frame_offset),
                    format!("Inf at element {idx}"),
                ));
                return;
            }
        }
    }
}

fn scan_bf16(
    data: &[u8],
    byte_order: ByteOrder,
    obj_idx: usize,
    frame_offset: usize,
    issues: &mut Vec<ValidationIssue>,
) {
    let chunks = data.chunks_exact(2);
    if !chunks.remainder().is_empty() {
        issues.push(err(
            IssueCode::DecodedSizeMismatch,
            ValidationLevel::Fidelity,
            Some(obj_idx),
            Some(frame_offset),
            format!(
                "decoded size {} is not a multiple of 2 (bfloat16)",
                data.len()
            ),
        ));
        return;
    }
    for (idx, chunk) in chunks.enumerate() {
        let bytes: [u8; 2] = chunk.try_into().unwrap();
        let bits = match byte_order {
            ByteOrder::Big => u16::from_be_bytes(bytes),
            ByteOrder::Little => u16::from_le_bytes(bytes),
        };
        // BFloat16: sign(1) + exponent(8) + mantissa(7)
        let exp = (bits >> 7) & 0xFF;
        let mantissa = bits & 0x7F;
        if exp == 0xFF {
            if mantissa != 0 {
                issues.push(err(
                    IssueCode::NanDetected,
                    ValidationLevel::Fidelity,
                    Some(obj_idx),
                    Some(frame_offset),
                    format!("NaN at element {idx}"),
                ));
                return;
            } else {
                issues.push(err(
                    IssueCode::InfDetected,
                    ValidationLevel::Fidelity,
                    Some(obj_idx),
                    Some(frame_offset),
                    format!("Inf at element {idx}"),
                ));
                return;
            }
        }
    }
}

fn scan_complex64(
    data: &[u8],
    byte_order: ByteOrder,
    obj_idx: usize,
    frame_offset: usize,
    issues: &mut Vec<ValidationIssue>,
) {
    // Complex64 = two f32 (real, imaginary) = 8 bytes per element
    let chunks = data.chunks_exact(8);
    if !chunks.remainder().is_empty() {
        issues.push(err(
            IssueCode::DecodedSizeMismatch,
            ValidationLevel::Fidelity,
            Some(obj_idx),
            Some(frame_offset),
            format!(
                "decoded size {} is not a multiple of 8 (complex64)",
                data.len()
            ),
        ));
        return;
    }
    for (idx, chunk) in chunks.enumerate() {
        let real_bytes: [u8; 4] = chunk[0..4].try_into().unwrap();
        let imag_bytes: [u8; 4] = chunk[4..8].try_into().unwrap();
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
        for (val, component) in [(real, "real"), (imag, "imaginary")] {
            if val.is_nan() {
                issues.push(err(
                    IssueCode::NanDetected,
                    ValidationLevel::Fidelity,
                    Some(obj_idx),
                    Some(frame_offset),
                    format!("NaN at element {idx} ({component} component)"),
                ));
                return;
            }
            if val.is_infinite() {
                issues.push(err(
                    IssueCode::InfDetected,
                    ValidationLevel::Fidelity,
                    Some(obj_idx),
                    Some(frame_offset),
                    format!("Inf at element {idx} ({component} component)"),
                ));
                return;
            }
        }
    }
}

fn scan_complex128(
    data: &[u8],
    byte_order: ByteOrder,
    obj_idx: usize,
    frame_offset: usize,
    issues: &mut Vec<ValidationIssue>,
) {
    // Complex128 = two f64 (real, imaginary) = 16 bytes per element
    let chunks = data.chunks_exact(16);
    if !chunks.remainder().is_empty() {
        issues.push(err(
            IssueCode::DecodedSizeMismatch,
            ValidationLevel::Fidelity,
            Some(obj_idx),
            Some(frame_offset),
            format!(
                "decoded size {} is not a multiple of 16 (complex128)",
                data.len()
            ),
        ));
        return;
    }
    for (idx, chunk) in chunks.enumerate() {
        let real_bytes: [u8; 8] = chunk[0..8].try_into().unwrap();
        let imag_bytes: [u8; 8] = chunk[8..16].try_into().unwrap();
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
        for (val, component) in [(real, "real"), (imag, "imaginary")] {
            if val.is_nan() {
                issues.push(err(
                    IssueCode::NanDetected,
                    ValidationLevel::Fidelity,
                    Some(obj_idx),
                    Some(frame_offset),
                    format!("NaN at element {idx} ({component} component)"),
                ));
                return;
            }
            if val.is_infinite() {
                issues.push(err(
                    IssueCode::InfDetected,
                    ValidationLevel::Fidelity,
                    Some(obj_idx),
                    Some(frame_offset),
                    format!("Inf at element {idx} ({component} component)"),
                ));
                return;
            }
        }
    }
}
