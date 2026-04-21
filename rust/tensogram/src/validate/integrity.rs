// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Level 3: Integrity validation — hash verification and decompression checks.

use crate::encode::build_pipeline_config;
use crate::error::TensogramError;
use crate::hash;
use crate::metadata;
use crate::wire::FrameType;

use super::structure::FrameWalkResult;
use super::types::*;

/// Result of a single hash check.
enum HashCheckResult {
    Verified,
    Skipped,
    Failed,
}

/// Verify the inline hash slot of the object frame at `frame_offset`.
///
/// Slices the full frame out of `buf` using the frame header's
/// `total_length`, then calls [`crate::hash::verify_frame_hash`]
/// which computes the xxh3-64 over the v3 hash scope
/// `bytes[16 .. total_length - footer_size(ft))` and compares it
/// against the u64 stored at `total_length - 12`.
fn verify_object_frame_hash(
    buf: &[u8],
    frame_offset: usize,
    obj_idx: usize,
    issues: &mut Vec<ValidationIssue>,
) -> HashCheckResult {
    // Read the frame header to learn the total length; if the
    // header doesn't parse, structure validation will have already
    // reported the issue — skip quietly here.
    if frame_offset + crate::wire::FRAME_HEADER_SIZE > buf.len() {
        return HashCheckResult::Skipped;
    }
    let fh = match crate::wire::FrameHeader::read_from(&buf[frame_offset..]) {
        Ok(fh) => fh,
        Err(_) => return HashCheckResult::Skipped,
    };
    let total = match usize::try_from(fh.total_length) {
        Ok(t) => t,
        Err(_) => return HashCheckResult::Skipped,
    };
    if frame_offset + total > buf.len() {
        return HashCheckResult::Skipped;
    }
    let frame_bytes = &buf[frame_offset..frame_offset + total];

    match hash::verify_frame_hash(frame_bytes, fh.frame_type) {
        Ok(()) => HashCheckResult::Verified,
        Err(TensogramError::HashMismatch { expected, actual }) => {
            issues.push(err(
                IssueCode::HashMismatch,
                ValidationLevel::Integrity,
                Some(obj_idx),
                Some(frame_offset),
                format!("frame hash mismatch (expected {expected}, got {actual})"),
            ));
            HashCheckResult::Failed
        }
        Err(e) => {
            issues.push(err(
                IssueCode::HashVerificationError,
                ValidationLevel::Integrity,
                Some(obj_idx),
                Some(frame_offset),
                format!("frame hash verification error: {e}"),
            ));
            HashCheckResult::Failed
        }
    }
}

/// `checksum_only`: skip decompression check, only verify hashes.
/// `cache_decoded`: retain decoded bytes in ObjectContext for Level 4 reuse.
///
/// **Memory note**: when `cache_decoded` is true (i.e. `--full`), decoded
/// payloads are retained for all non-raw objects until Level 4 completes.
/// Peak memory is proportional to the sum of decoded object sizes, not
/// the message size. For very large tensors, consider validating objects
/// individually or using `--checksum` for quick integrity checks.
pub(crate) fn validate_integrity(
    buf: &[u8],
    walk: &FrameWalkResult<'_>,
    objects: &mut [ObjectContext<'_>],
    issues: &mut Vec<ValidationIssue>,
    checksum_only: bool,
    cache_decoded: bool,
) -> bool {
    let mut all_verified = true;
    let mut any_checked = false;

    // Parse aggregate HashFrame(s) (HeaderHash / FooterHash) and
    // surface any structural or schema-level issues.  v3 only
    // recognises `algorithm = "xxh3"`; other values trigger an
    // `UnknownHashAlgorithm` warning — the frame-body hashes are
    // still verified against the inline slots below (which are
    // authoritative), but an unknown algorithm name means the
    // aggregate HashFrame entries can't be interpreted.
    for (ft, payload) in &walk.meta_frames {
        if !matches!(ft, FrameType::HeaderHash | FrameType::FooterHash) {
            continue;
        }
        match metadata::cbor_to_hash_frame(payload) {
            Ok(hf) => {
                if hash::HashAlgorithm::parse(&hf.algorithm).is_err() {
                    issues.push(warn(
                        IssueCode::UnknownHashAlgorithm,
                        ValidationLevel::Integrity,
                        None,
                        None,
                        format!(
                            "unknown hash algorithm '{}' in HashFrame, aggregate \
                             entries cannot be verified (inline slots still checked)",
                            hf.algorithm
                        ),
                    ));
                }
            }
            Err(e) => {
                all_verified = false;
                issues.push(err(
                    IssueCode::HashFrameCborParseFailed,
                    ValidationLevel::Integrity,
                    None,
                    None,
                    format!("failed to parse hash frame CBOR: {e}"),
                ));
            }
        }
    }

    // Check whether the preamble declares any hashes at all.  If
    // `HASHES_PRESENT` is 0, every frame's inline slot is zero and
    // we skip verification with a clear message.
    let has_hashes = crate::wire::Preamble::read_from(buf)
        .map(|p| p.flags.has(crate::wire::MessageFlags::HASHES_PRESENT))
        .unwrap_or(false);

    for (i, obj) in objects.iter_mut().enumerate() {
        // If Level 2 didn't run, try parsing the descriptor now.
        // If Level 2 already tried and failed, skip (don't re-parse or duplicate errors).
        if obj.descriptor.is_none() && !obj.descriptor_failed {
            match metadata::cbor_to_object_descriptor(obj.cbor_bytes) {
                Ok(d) => {
                    obj.descriptor = Some(d);
                }
                Err(e) => {
                    obj.descriptor_failed = true;
                    issues.push(err(
                        IssueCode::HashVerificationError,
                        ValidationLevel::Integrity,
                        Some(i),
                        None,
                        format!("failed to parse descriptor: {e}"),
                    ));
                }
            }
        }

        // v3 integrity: verify the inline hash slot at
        // `frame_start + total_length - 12`.  Slice out the full
        // frame from `buf` using the frame offset and header.
        let result = if !has_hashes {
            // Message-level HASHES_PRESENT = 0 — no hash to verify.
            issues.push(warn(
                IssueCode::NoHashAvailable,
                ValidationLevel::Integrity,
                Some(i),
                None,
                "HASHES_PRESENT flag is clear — message was encoded \
                 without hashes; cannot verify integrity"
                    .to_string(),
            ));
            all_verified = false;
            HashCheckResult::Skipped
        } else {
            match verify_object_frame_hash(buf, obj.frame_offset, i, issues) {
                HashCheckResult::Verified => {
                    any_checked = true;
                    HashCheckResult::Verified
                }
                other => other,
            }
        };

        if !matches!(result, HashCheckResult::Verified) {
            all_verified = false;
        }

        // Decompression check (requires parsed descriptor, skipped in checksum-only mode)
        if !checksum_only
            && let Some(ref desc) = obj.descriptor
            && (desc.compression != "none" || desc.encoding != "none" || desc.filter != "none")
        {
            let shape_product = desc
                .shape
                .iter()
                .try_fold(1u64, |acc, &x| acc.checked_mul(x));
            let num_elements = match shape_product {
                Some(product) => match usize::try_from(product) {
                    Ok(n) => Some(n),
                    Err(_) => {
                        obj.decode_state = DecodeState::DecodeFailed;
                        issues.push(err(
                            IssueCode::PipelineConfigFailed,
                            ValidationLevel::Integrity,
                            Some(i),
                            Some(obj.frame_offset),
                            format!("shape product {} does not fit in usize", product),
                        ));
                        None
                    }
                },
                None => {
                    // Shape overflow already reported at Level 2
                    obj.decode_state = DecodeState::DecodeFailed;
                    None
                }
            };
            if let Some(num_elements) = num_elements {
                match build_pipeline_config(desc, num_elements, desc.dtype) {
                    Ok(config) => {
                        match tensogram_encodings::pipeline::decode_pipeline(
                            obj.payload,
                            &config,
                            false,
                        ) {
                            Ok(decoded) => {
                                if cache_decoded {
                                    obj.decode_state = DecodeState::Decoded(decoded);
                                }
                            }
                            Err(e) => {
                                obj.decode_state = DecodeState::DecodeFailed;
                                issues.push(err(
                                    IssueCode::DecodePipelineFailed,
                                    ValidationLevel::Integrity,
                                    Some(i),
                                    Some(obj.frame_offset),
                                    format!("decode pipeline failed: {e}"),
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        obj.decode_state = DecodeState::DecodeFailed;
                        issues.push(err(
                            IssueCode::PipelineConfigFailed,
                            ValidationLevel::Integrity,
                            Some(i),
                            Some(obj.frame_offset),
                            format!("cannot build pipeline config: {e}"),
                        ));
                    }
                }
            }
        }
    }

    any_checked && all_verified
}
