//! Level 2: Metadata validation — CBOR parsing and consistency checks.

use crate::metadata;
use crate::types::{DataObjectDescriptor, GlobalMetadata};
use crate::wire::FrameType;

use super::structure::FrameWalkResult;
use super::types::*;

pub(crate) fn validate_metadata(
    walk: &FrameWalkResult<'_>,
    issues: &mut Vec<ValidationIssue>,
    check_canonical: bool,
) {
    let mut global_meta: Option<GlobalMetadata> = None;
    let mut meta_base_len_before_normalization: Option<usize> = None;

    for (ft, payload) in &walk.meta_frames {
        match ft {
            FrameType::HeaderMetadata | FrameType::FooterMetadata => {
                if check_canonical {
                    if let Err(e) = metadata::verify_canonical_cbor(payload) {
                        issues.push(warn(
                            IssueCode::MetadataCborNonCanonical,
                            ValidationLevel::Metadata,
                            None,
                            None,
                            format!("metadata CBOR is not canonical: {e}"),
                        ));
                    }
                }
                match metadata::cbor_to_global_metadata(payload) {
                    Ok(meta) => {
                        meta_base_len_before_normalization = Some(meta.base.len());
                        global_meta = Some(meta);
                    }
                    Err(e) => {
                        issues.push(err(
                            IssueCode::MetadataCborParseFailed,
                            ValidationLevel::Metadata,
                            None,
                            None,
                            format!("failed to parse metadata CBOR: {e}"),
                        ));
                        return;
                    }
                }
            }
            FrameType::HeaderIndex | FrameType::FooterIndex => {
                match metadata::cbor_to_index(payload) {
                    Ok(idx) => {
                        let obj_count = walk.data_objects.len();
                        let indexed_count = usize::try_from(idx.object_count).unwrap_or(usize::MAX);
                        if indexed_count != obj_count {
                            issues.push(err(
                                IssueCode::IndexCountMismatch,
                                ValidationLevel::Metadata,
                                None,
                                None,
                                format!(
                                    "index object_count {} != actual data object count {}",
                                    idx.object_count, obj_count
                                ),
                            ));
                        }
                        if idx.offsets.len() != obj_count {
                            issues.push(err(
                                IssueCode::IndexCountMismatch,
                                ValidationLevel::Metadata,
                                None,
                                None,
                                format!(
                                    "index offsets length {} != data object count {}",
                                    idx.offsets.len(),
                                    obj_count
                                ),
                            ));
                        }
                        // Verify each index offset points to the actual data object
                        for (j, &idx_offset) in idx.offsets.iter().enumerate() {
                            if j < walk.data_objects.len() {
                                let actual_offset = walk.data_objects[j].2;
                                let offset_matches = usize::try_from(idx_offset)
                                    .map(|o| o == actual_offset)
                                    .unwrap_or(false);
                                if !offset_matches {
                                    issues.push(err(
                                        IssueCode::IndexOffsetMismatch,
                                        ValidationLevel::Metadata,
                                        Some(j),
                                        None,
                                        format!(
                                            "index offset[{j}] = {} but actual data object frame at {}",
                                            idx_offset, actual_offset
                                        ),
                                    ));
                                }
                            }
                        }
                    }
                    Err(e) => {
                        issues.push(err(
                            IssueCode::IndexCborParseFailed,
                            ValidationLevel::Metadata,
                            None,
                            None,
                            format!("failed to parse index CBOR: {e}"),
                        ));
                    }
                }
            }
            FrameType::HeaderHash | FrameType::FooterHash => {
                match metadata::cbor_to_hash_frame(payload) {
                    Ok(hf) => {
                        let obj_count = walk.data_objects.len();
                        if hf.hashes.len() != obj_count {
                            issues.push(err(
                                IssueCode::HashFrameCountMismatch,
                                ValidationLevel::Metadata,
                                None,
                                None,
                                format!(
                                    "hash frame has {} hashes but {} data objects",
                                    hf.hashes.len(),
                                    obj_count
                                ),
                            ));
                        }
                    }
                    Err(e) => {
                        issues.push(err(
                            IssueCode::HashFrameCborParseFailed,
                            ValidationLevel::Metadata,
                            None,
                            None,
                            format!("failed to parse hash frame CBOR: {e}"),
                        ));
                    }
                }
            }
            _ => {}
        }
    }

    // Validate preceder metadata frames
    for (ft, payload) in &walk.meta_frames {
        if *ft == FrameType::PrecederMetadata {
            if check_canonical {
                if let Err(e) = metadata::verify_canonical_cbor(payload) {
                    issues.push(warn(
                        IssueCode::PrecederCborNonCanonical,
                        ValidationLevel::Metadata,
                        None,
                        None,
                        format!("preceder metadata CBOR is not canonical: {e}"),
                    ));
                }
            }
            match metadata::cbor_to_global_metadata(payload) {
                Ok(prec) => {
                    if prec.base.len() != 1 {
                        issues.push(err(
                            IssueCode::PrecederBaseCountWrong,
                            ValidationLevel::Metadata,
                            None,
                            None,
                            format!(
                                "PrecederMetadata base must have exactly 1 entry, got {}",
                                prec.base.len()
                            ),
                        ));
                    }
                }
                Err(e) => {
                    issues.push(err(
                        IssueCode::PrecederCborParseFailed,
                        ValidationLevel::Metadata,
                        None,
                        None,
                        format!("failed to parse preceder metadata CBOR: {e}"),
                    ));
                }
            }
        }
    }

    let meta = match global_meta {
        Some(m) => m,
        None => return,
    };

    // base.len() vs object count (before normalization)
    let obj_count = walk.data_objects.len();
    if let Some(base_len) = meta_base_len_before_normalization {
        if base_len > obj_count {
            issues.push(err(
                IssueCode::BaseCountExceedsObjects,
                ValidationLevel::Metadata,
                None,
                None,
                format!(
                    "metadata base has {} entries but message has {} data objects",
                    base_len, obj_count
                ),
            ));
        }
    }

    // Per-object descriptor validation
    for (i, (cbor_bytes, _payload, _offset)) in walk.data_objects.iter().enumerate() {
        if check_canonical {
            if let Err(e) = metadata::verify_canonical_cbor(cbor_bytes) {
                issues.push(warn(
                    IssueCode::DescriptorCborNonCanonical,
                    ValidationLevel::Metadata,
                    Some(i),
                    None,
                    format!("descriptor CBOR is not canonical: {e}"),
                ));
            }
        }

        let desc: DataObjectDescriptor = match metadata::cbor_to_object_descriptor(cbor_bytes) {
            Ok(d) => d,
            Err(e) => {
                issues.push(err(
                    IssueCode::DescriptorCborParseFailed,
                    ValidationLevel::Metadata,
                    Some(i),
                    None,
                    format!("descriptor CBOR parse failed: {e}"),
                ));
                continue;
            }
        };

        if desc.ndim as usize != desc.shape.len() {
            issues.push(err(
                IssueCode::NdimShapeMismatch,
                ValidationLevel::Metadata,
                Some(i),
                None,
                format!("ndim {} != shape.len() {}", desc.ndim, desc.shape.len()),
            ));
        }
        if desc.strides.len() != desc.shape.len() {
            issues.push(err(
                IssueCode::StridesShapeMismatch,
                ValidationLevel::Metadata,
                Some(i),
                None,
                format!(
                    "strides.len() {} != shape.len() {}",
                    desc.strides.len(),
                    desc.shape.len()
                ),
            ));
        }
        if desc
            .shape
            .iter()
            .try_fold(1u64, |acc, &x| acc.checked_mul(x))
            .is_none()
        {
            issues.push(err(
                IssueCode::ShapeOverflow,
                ValidationLevel::Metadata,
                Some(i),
                None,
                "shape product overflows u64".to_string(),
            ));
        }

        if !matches!(desc.encoding.as_str(), "none" | "simple_packing") {
            issues.push(err(
                IssueCode::UnknownEncoding,
                ValidationLevel::Metadata,
                Some(i),
                None,
                format!("unknown encoding '{}'", desc.encoding),
            ));
        }
        if !matches!(desc.filter.as_str(), "none" | "shuffle") {
            issues.push(err(
                IssueCode::UnknownFilter,
                ValidationLevel::Metadata,
                Some(i),
                None,
                format!("unknown filter '{}'", desc.filter),
            ));
        }
        // All codecs defined in the wire format, regardless of build features.
        // Level 2 checks format validity; Level 3 catches unsupported codecs
        // at decode time via PipelineConfigFailed.
        let known_compressions = ["none", "szip", "zstd", "lz4", "blosc2", "zfp", "sz3"];
        if !known_compressions.contains(&desc.compression.as_str()) {
            issues.push(err(
                IssueCode::UnknownCompression,
                ValidationLevel::Metadata,
                Some(i),
                None,
                format!("unknown compression '{}'", desc.compression),
            ));
        }

        if desc.obj_type.is_empty() {
            issues.push(err(
                IssueCode::EmptyObjType,
                ValidationLevel::Metadata,
                Some(i),
                None,
                "obj_type is empty".to_string(),
            ));
        }
    }

    // Validate _reserved_.tensor in each base entry
    for (i, entry) in meta.base.iter().enumerate().take(obj_count) {
        if let Some(reserved) = entry.get("_reserved_") {
            if let ciborium::Value::Map(pairs) = reserved {
                let has_tensor = pairs
                    .iter()
                    .any(|(k, _)| matches!(k, ciborium::Value::Text(s) if s == "tensor"));
                if !has_tensor {
                    issues.push(warn(
                        IssueCode::ReservedMissingTensor,
                        ValidationLevel::Metadata,
                        Some(i),
                        None,
                        "base._reserved_ missing 'tensor' key".to_string(),
                    ));
                }
            } else {
                issues.push(err(
                    IssueCode::ReservedNotAMap,
                    ValidationLevel::Metadata,
                    Some(i),
                    None,
                    "base._reserved_ is not a map".to_string(),
                ));
            }
        }
    }
}
