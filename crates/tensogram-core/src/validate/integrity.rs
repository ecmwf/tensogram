//! Level 3: Integrity validation — hash verification and decompression checks.

use crate::encode::build_pipeline_config;
use crate::error::TensogramError;
use crate::hash;
use crate::metadata;
use crate::types::DataObjectDescriptor;
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
            format!(
                "object {obj_idx}: unknown hash algorithm '{}', cannot verify",
                h.hash_type
            ),
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
                format!("object {obj_idx}: hash mismatch (expected {expected}, got {actual})"),
            ));
            HashCheckResult::Failed
        }
        Err(e) => {
            issues.push(err(
                IssueCode::HashVerificationError,
                ValidationLevel::Integrity,
                Some(obj_idx),
                None,
                format!("object {obj_idx}: hash verification error: {e}"),
            ));
            HashCheckResult::Failed
        }
    }
}

pub(crate) fn validate_integrity(
    walk: &FrameWalkResult<'_>,
    issues: &mut Vec<ValidationIssue>,
) -> bool {
    let mut all_verified = true;
    let mut any_checked = false;

    // Collect hash frame if present
    let mut hash_frame: Option<crate::types::HashFrame> = None;
    for (ft, payload) in &walk.meta_frames {
        if matches!(ft, FrameType::HeaderHash | FrameType::FooterHash) {
            if let Ok(hf) = metadata::cbor_to_hash_frame(payload) {
                hash_frame = Some(hf);
            }
        }
    }

    for (i, (cbor_bytes, payload, _offset)) in walk.data_objects.iter().enumerate() {
        let desc: DataObjectDescriptor = match metadata::cbor_to_object_descriptor(cbor_bytes) {
            Ok(d) => d,
            Err(_) => {
                all_verified = false;
                continue;
            }
        };

        // Hash verification: prefer per-object descriptor hash, fall back to hash frame
        let result = if let Some(ref h) = desc.hash {
            any_checked = true;
            check_hash(payload, h, i, issues)
        } else if let Some(ref hf) = hash_frame {
            if i < hf.hashes.len() {
                any_checked = true;
                let h = crate::types::HashDescriptor {
                    hash_type: hf.hash_type.clone(),
                    value: hf.hashes[i].clone(),
                };
                check_hash(payload, &h, i, issues)
            } else {
                issues.push(warn(
                    IssueCode::NoHashAvailable,
                    ValidationLevel::Integrity,
                    Some(i),
                    None,
                    format!("object {i}: no hash available, cannot verify integrity"),
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
                format!("object {i}: no hash available, cannot verify integrity"),
            ));
            all_verified = false;
            HashCheckResult::Skipped
        };

        if !matches!(result, HashCheckResult::Verified) {
            all_verified = false;
        }

        // Decompression check
        if desc.compression != "none" || desc.encoding != "none" || desc.filter != "none" {
            let shape_product = desc
                .shape
                .iter()
                .try_fold(1u64, |acc, &x| acc.checked_mul(x));
            if let Some(product) = shape_product {
                if let Ok(num_elements) = usize::try_from(product) {
                    match build_pipeline_config(&desc, num_elements, desc.dtype) {
                        Ok(config) => {
                            if let Err(e) =
                                tensogram_encodings::pipeline::decode_pipeline(payload, &config)
                            {
                                issues.push(err(
                                    IssueCode::DecodePipelineFailed,
                                    ValidationLevel::Integrity,
                                    Some(i),
                                    None,
                                    format!("object {i}: decode pipeline failed: {e}"),
                                ));
                            }
                        }
                        Err(e) => {
                            issues.push(err(
                                IssueCode::PipelineConfigFailed,
                                ValidationLevel::Integrity,
                                Some(i),
                                None,
                                format!("object {i}: cannot build pipeline config: {e}"),
                            ));
                        }
                    }
                }
            }
        }
    }

    any_checked && all_verified
}
