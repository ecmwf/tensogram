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

/// Check a single hash descriptor against payload, pushing issues on failure.
fn check_hash(
    payload: &[u8],
    h: &crate::types::HashDescriptor,
    obj_idx: usize,
    issues: &mut Vec<ValidationIssue>,
) -> HashCheckResult {
    if hash::HashAlgorithm::parse(&h.hash_type).is_err() {
        issues.push(warn(
            IssueCode::UnknownHashAlgorithm,
            ValidationLevel::Integrity,
            Some(obj_idx),
            None,
            format!("unknown hash algorithm '{}', cannot verify", h.hash_type),
        ));
        return HashCheckResult::Skipped;
    }
    match hash::verify_hash(payload, h) {
        Ok(()) => HashCheckResult::Verified,
        Err(TensogramError::HashMismatch { expected, actual }) => {
            issues.push(err(
                IssueCode::HashMismatch,
                ValidationLevel::Integrity,
                Some(obj_idx),
                None,
                format!("hash mismatch (expected {expected}, got {actual})"),
            ));
            HashCheckResult::Failed
        }
        Err(e) => {
            issues.push(err(
                IssueCode::HashVerificationError,
                ValidationLevel::Integrity,
                Some(obj_idx),
                None,
                format!("hash verification error: {e}"),
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
    walk: &FrameWalkResult<'_>,
    objects: &mut [ObjectContext<'_>],
    issues: &mut Vec<ValidationIssue>,
    checksum_only: bool,
    cache_decoded: bool,
) -> bool {
    let mut all_verified = true;
    let mut any_checked = false;

    // Collect hash frame if present
    let mut hash_frame: Option<crate::types::HashFrame> = None;
    for (ft, payload) in &walk.meta_frames {
        if matches!(ft, FrameType::HeaderHash | FrameType::FooterHash) {
            match metadata::cbor_to_hash_frame(payload) {
                Ok(hf) => {
                    hash_frame = Some(hf);
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
    }

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
                        format!("failed to parse descriptor, falling back to hash frame: {e}"),
                    ));
                }
            }
        }

        // Hash verification: prefer per-object descriptor hash, fall back to hash frame
        let result = if let Some(h) = obj.descriptor.as_ref().and_then(|d| d.hash.as_ref()) {
            any_checked = true;
            check_hash(obj.payload, h, i, issues)
        } else if let Some(ref hf) = hash_frame {
            if i < hf.hashes.len() {
                any_checked = true;
                let h = crate::types::HashDescriptor {
                    hash_type: hf.hash_type.clone(),
                    value: hf.hashes[i].clone(),
                };
                check_hash(obj.payload, &h, i, issues)
            } else {
                issues.push(warn(
                    IssueCode::NoHashAvailable,
                    ValidationLevel::Integrity,
                    Some(i),
                    None,
                    "no hash available, cannot verify integrity".to_string(),
                ));
                all_verified = false;
                HashCheckResult::Skipped
            }
        } else {
            issues.push(warn(
                IssueCode::NoHashAvailable,
                ValidationLevel::Integrity,
                Some(i),
                None,
                "no hash available, cannot verify integrity".to_string(),
            ));
            all_verified = false;
            HashCheckResult::Skipped
        };

        if !matches!(result, HashCheckResult::Verified) {
            all_verified = false;
        }

        // Decompression check (requires parsed descriptor, skipped in checksum-only mode)
        if !checksum_only {
            if let Some(ref desc) = obj.descriptor {
                if desc.compression != "none" || desc.encoding != "none" || desc.filter != "none" {
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
        } // if !checksum_only
    }

    any_checked && all_verified
}
