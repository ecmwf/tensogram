//! Validation of tensogram messages and files.
//!
//! Provides `validate_message()` for checking a single message buffer and
//! `validate_file()` for checking all messages in a `.tgm` file, including
//! detection of truncated or garbage bytes between messages.

use std::path::Path;

use crate::encode::build_pipeline_config;
use crate::error::TensogramError;
use crate::hash;
use crate::metadata;
use crate::types::{DataObjectDescriptor, GlobalMetadata};
use crate::wire::{
    self, DataObjectFlags, FrameHeader, FrameType, MessageFlags, Postamble, Preamble,
    DATA_OBJECT_FOOTER_SIZE, FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC, MAGIC, POSTAMBLE_SIZE,
    PREAMBLE_SIZE,
};

// ── Public types ────────────────────────────────────────────────────────────

/// Validation levels, from lightest to most thorough.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationLevel {
    /// Level 1: magic bytes, frame structure, lengths, ordering.
    Structure = 1,
    /// Level 2: CBOR parses, required keys present, types recognized.
    Metadata = 2,
    /// Level 3: hash verification, decompression without value interpretation.
    Integrity = 3,
}

/// How to run validation — selects which levels are included.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidateMode {
    /// Level 1 only (--quick).
    Quick,
    /// Levels 1–3 (default).
    Default,
    /// Level 3 only (--checksum).
    Checksum,
    /// Levels 1–3 plus opt-in canonical CBOR check.
    Canonical,
}

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    Error,
    Warning,
}

/// A single validation finding.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub level: ValidationLevel,
    pub severity: IssueSeverity,
    /// Index of the object within the message (if applicable).
    pub object_index: Option<usize>,
    /// Byte offset within the message buffer (if applicable).
    pub byte_offset: Option<usize>,
    pub description: String,
}

/// Options passed to `validate_message`.
#[derive(Debug, Clone)]
pub struct ValidateOptions {
    pub mode: ValidateMode,
}

impl Default for ValidateOptions {
    fn default() -> Self {
        Self {
            mode: ValidateMode::Default,
        }
    }
}

/// Result of validating a single message.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub issues: Vec<ValidationIssue>,
    pub object_count: usize,
    pub hash_verified: bool,
}

impl ValidationReport {
    pub fn is_ok(&self) -> bool {
        !self
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Error)
    }
}

/// A file-level issue (not tied to a specific message).
#[derive(Debug, Clone)]
pub struct FileIssue {
    pub byte_offset: usize,
    pub length: usize,
    pub description: String,
}

/// Result of validating a `.tgm` file.
#[derive(Debug, Clone)]
pub struct FileValidationReport {
    /// Issues at the file level (gaps, trailing bytes, truncated messages).
    pub file_issues: Vec<FileIssue>,
    /// Per-message validation reports.
    pub messages: Vec<ValidationReport>,
}

impl FileValidationReport {
    pub fn is_ok(&self) -> bool {
        self.file_issues.is_empty() && self.messages.iter().all(|r| r.is_ok())
    }

    pub fn total_objects(&self) -> usize {
        self.messages.iter().map(|r| r.object_count).sum()
    }

    pub fn hash_verified(&self) -> bool {
        !self.messages.is_empty() && self.messages.iter().all(|r| r.hash_verified)
    }
}

// ── Internal helpers ────────────────────────────────────────────────────────

fn issue(
    level: ValidationLevel,
    severity: IssueSeverity,
    object_index: Option<usize>,
    byte_offset: Option<usize>,
    description: impl Into<String>,
) -> ValidationIssue {
    ValidationIssue {
        level,
        severity,
        object_index,
        byte_offset,
        description: description.into(),
    }
}

fn err(
    level: ValidationLevel,
    object_index: Option<usize>,
    byte_offset: Option<usize>,
    description: impl Into<String>,
) -> ValidationIssue {
    issue(
        level,
        IssueSeverity::Error,
        object_index,
        byte_offset,
        description,
    )
}

fn warn(
    level: ValidationLevel,
    object_index: Option<usize>,
    byte_offset: Option<usize>,
    description: impl Into<String>,
) -> ValidationIssue {
    issue(
        level,
        IssueSeverity::Warning,
        object_index,
        byte_offset,
        description,
    )
}

// ── Frame walking state (Level 1) ───────────────────────────────────────────

/// Phases for frame ordering enforcement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Phase {
    Headers,
    DataObjects,
    Footers,
}

fn frame_phase(ft: FrameType) -> Phase {
    match ft {
        FrameType::HeaderMetadata | FrameType::HeaderIndex | FrameType::HeaderHash => {
            Phase::Headers
        }
        FrameType::DataObject | FrameType::PrecederMetadata => Phase::DataObjects,
        FrameType::FooterHash | FrameType::FooterIndex | FrameType::FooterMetadata => {
            Phase::Footers
        }
    }
}

/// Info collected during Level 1 frame walk, reused by Level 2/3.
struct FrameWalkResult<'a> {
    /// (frame_type, payload_bytes) for non-DataObject frames.
    meta_frames: Vec<(FrameType, &'a [u8])>,
    /// (descriptor_cbor_bytes, payload_bytes, frame_start_offset) per data object.
    data_objects: Vec<(Vec<u8>, &'a [u8], usize)>,
}

// ── Level 1: Structure ──────────────────────────────────────────────────────

/// Walk the raw bytes of a message, collecting structural issues.
/// Returns the walk result (for reuse by Level 2/3) or None if structure
/// is too broken to continue.
fn validate_structure<'a>(
    buf: &'a [u8],
    issues: &mut Vec<ValidationIssue>,
) -> Option<FrameWalkResult<'a>> {
    // --- Preamble ---
    if buf.len() < PREAMBLE_SIZE {
        issues.push(err(
            ValidationLevel::Structure,
            None,
            Some(0),
            format!(
                "buffer too short for preamble: {} < {PREAMBLE_SIZE}",
                buf.len()
            ),
        ));
        return None;
    }

    if &buf[0..8] != MAGIC {
        issues.push(err(
            ValidationLevel::Structure,
            None,
            Some(0),
            format!(
                "invalid magic bytes: expected TENSOGRM, got {:?}",
                String::from_utf8_lossy(&buf[0..8])
            ),
        ));
        return None;
    }

    let preamble = match Preamble::read_from(buf) {
        Ok(p) => p,
        Err(e) => {
            issues.push(err(
                ValidationLevel::Structure,
                None,
                Some(0),
                format!("preamble parse failed: {e}"),
            ));
            return None;
        }
    };

    // --- Total length / postamble ---
    let msg_end = if preamble.total_length > 0 {
        let total = match usize::try_from(preamble.total_length) {
            Ok(t) => t,
            Err(_) => {
                issues.push(err(
                    ValidationLevel::Structure,
                    None,
                    Some(16),
                    format!("total_length {} overflows usize", preamble.total_length),
                ));
                return None;
            }
        };
        if total > buf.len() {
            issues.push(err(
                ValidationLevel::Structure,
                None,
                Some(16),
                format!("total_length {} exceeds buffer size {}", total, buf.len()),
            ));
            return None;
        }
        let min_msg_size = PREAMBLE_SIZE + POSTAMBLE_SIZE;
        if total < min_msg_size {
            issues.push(err(
                ValidationLevel::Structure,
                None,
                Some(16),
                format!(
                    "total_length {} is smaller than minimum message size {min_msg_size}",
                    total
                ),
            ));
            return None;
        }
        // Validate postamble
        let pa_offset = total - POSTAMBLE_SIZE;
        match Postamble::read_from(&buf[pa_offset..]) {
            Ok(pa) => {
                // Validate first_footer_offset is within message
                let ffo = pa.first_footer_offset as usize;
                if ffo < PREAMBLE_SIZE || ffo > pa_offset {
                    issues.push(err(
                        ValidationLevel::Structure,
                        None,
                        Some(pa_offset),
                        format!(
                            "first_footer_offset {} out of range [{PREAMBLE_SIZE}, {pa_offset}]",
                            ffo
                        ),
                    ));
                }
            }
            Err(e) => {
                issues.push(err(
                    ValidationLevel::Structure,
                    None,
                    Some(pa_offset),
                    format!("postamble invalid: {e}"),
                ));
                return None;
            }
        }
        total - POSTAMBLE_SIZE
    } else {
        // Streaming mode: postamble at end of buffer
        if buf.len() < POSTAMBLE_SIZE {
            issues.push(err(
                ValidationLevel::Structure,
                None,
                Some(0),
                "buffer too short for postamble in streaming mode".to_string(),
            ));
            return None;
        }
        let pa_offset = buf.len() - POSTAMBLE_SIZE;
        if Postamble::read_from(&buf[pa_offset..]).is_err() {
            issues.push(err(
                ValidationLevel::Structure,
                None,
                Some(pa_offset),
                "missing or invalid postamble in streaming message".to_string(),
            ));
            return None;
        }
        pa_offset
    };

    // --- Frame walk ---
    let mut pos = PREAMBLE_SIZE;
    let mut current_phase = Phase::Headers;
    let mut meta_frames: Vec<(FrameType, &[u8])> = Vec::new();
    let mut data_objects: Vec<(Vec<u8>, &[u8], usize)> = Vec::new();
    let mut observed_flags = MessageFlags::new(0);
    let mut pending_preceder = false;
    let mut obj_idx: usize = 0;

    while pos < msg_end {
        // Skip padding bytes between frames (alignment to 8-byte boundary)
        if pos + 2 > msg_end {
            // Remaining bytes are padding
            break;
        }
        if &buf[pos..pos + 2] != FRAME_MAGIC {
            // Could be padding
            pos += 1;
            continue;
        }

        if pos + FRAME_HEADER_SIZE > msg_end {
            issues.push(err(
                ValidationLevel::Structure,
                None,
                Some(pos),
                format!("truncated frame header at offset {pos}"),
            ));
            break;
        }

        let fh = match FrameHeader::read_from(&buf[pos..]) {
            Ok(fh) => fh,
            Err(e) => {
                issues.push(err(
                    ValidationLevel::Structure,
                    None,
                    Some(pos),
                    format!("invalid frame header at offset {pos}: {e}"),
                ));
                break;
            }
        };

        let frame_total = match usize::try_from(fh.total_length) {
            Ok(t) => t,
            Err(_) => {
                issues.push(err(
                    ValidationLevel::Structure,
                    None,
                    Some(pos),
                    format!(
                        "frame total_length {} overflows usize at offset {pos}",
                        fh.total_length
                    ),
                ));
                break;
            }
        };

        let min_size = if fh.frame_type == FrameType::DataObject {
            FRAME_HEADER_SIZE + DATA_OBJECT_FOOTER_SIZE
        } else {
            FRAME_HEADER_SIZE + FRAME_END.len()
        };
        if frame_total < min_size {
            issues.push(err(
                ValidationLevel::Structure,
                None,
                Some(pos),
                format!(
                    "frame at offset {pos} total_length {} < minimum {min_size}",
                    frame_total
                ),
            ));
            break;
        }

        let frame_end = match pos.checked_add(frame_total) {
            Some(end) => end,
            None => {
                issues.push(err(
                    ValidationLevel::Structure,
                    None,
                    Some(pos),
                    format!("frame at offset {pos} total_length {frame_total} causes overflow"),
                ));
                break;
            }
        };
        if frame_end > msg_end.saturating_add(POSTAMBLE_SIZE) {
            issues.push(err(
                ValidationLevel::Structure,
                None,
                Some(pos),
                format!(
                    "frame at offset {pos} extends beyond message (frame_end={frame_end}, msg_end={})",
                    msg_end + POSTAMBLE_SIZE
                ),
            ));
            break;
        }

        // Validate ENDF marker
        let endf_offset = pos + frame_total - FRAME_END.len();
        if &buf[endf_offset..endf_offset + FRAME_END.len()] != FRAME_END {
            issues.push(err(
                ValidationLevel::Structure,
                None,
                Some(endf_offset),
                format!("missing ENDF marker at offset {endf_offset} for frame starting at {pos}"),
            ));
            break;
        }

        // Frame ordering
        let phase = frame_phase(fh.frame_type);
        if phase < current_phase {
            issues.push(err(
                ValidationLevel::Structure,
                None,
                Some(pos),
                format!(
                    "frame {:?} at offset {pos} appears after {:?} phase — wrong order",
                    fh.frame_type, current_phase
                ),
            ));
        }
        current_phase = phase;

        // Preceder legality
        if pending_preceder && fh.frame_type != FrameType::DataObject {
            issues.push(err(
                ValidationLevel::Structure,
                None,
                Some(pos),
                format!(
                    "PrecederMetadata must be followed by DataObject, got {:?} at offset {pos}",
                    fh.frame_type
                ),
            ));
            pending_preceder = false;
        }

        // Track observed flags
        match fh.frame_type {
            FrameType::HeaderMetadata => observed_flags.set(MessageFlags::HEADER_METADATA),
            FrameType::FooterMetadata => observed_flags.set(MessageFlags::FOOTER_METADATA),
            FrameType::HeaderIndex => observed_flags.set(MessageFlags::HEADER_INDEX),
            FrameType::FooterIndex => observed_flags.set(MessageFlags::FOOTER_INDEX),
            FrameType::HeaderHash => observed_flags.set(MessageFlags::HEADER_HASHES),
            FrameType::FooterHash => observed_flags.set(MessageFlags::FOOTER_HASHES),
            FrameType::PrecederMetadata => {
                pending_preceder = true;
                observed_flags.set(MessageFlags::PRECEDER_METADATA);
            }
            FrameType::DataObject => {
                pending_preceder = false;
            }
        }

        // Extract payloads for Level 2/3 reuse
        let frame_start = pos;
        match fh.frame_type {
            FrameType::DataObject => {
                // Extract descriptor CBOR and payload from data object frame
                let cbor_offset_pos = endf_offset - 8;
                if cbor_offset_pos < pos + FRAME_HEADER_SIZE {
                    issues.push(err(
                        ValidationLevel::Structure,
                        Some(obj_idx),
                        Some(pos),
                        format!("data object frame at offset {pos} too small for cbor_offset"),
                    ));
                } else {
                    let cbor_offset_raw = wire::read_u64_be(buf, cbor_offset_pos);
                    let cbor_offset = cbor_offset_raw as usize;
                    let abs_cbor_offset = pos + cbor_offset;

                    if cbor_offset < FRAME_HEADER_SIZE || abs_cbor_offset > cbor_offset_pos {
                        issues.push(err(
                            ValidationLevel::Structure,
                            Some(obj_idx),
                            Some(pos),
                            format!(
                                "data object cbor_offset {} out of range at offset {pos}",
                                cbor_offset
                            ),
                        ));
                    } else {
                        let cbor_after = fh.flags & DataObjectFlags::CBOR_AFTER_PAYLOAD != 0;
                        let (cbor_slice, payload_slice) = if cbor_after {
                            let payload_start = pos + FRAME_HEADER_SIZE;
                            let cbor_start = abs_cbor_offset;
                            let cbor_end = cbor_offset_pos;
                            (&buf[cbor_start..cbor_end], &buf[payload_start..cbor_start])
                        } else {
                            // CBOR-before layout: header(16) | cbor | payload | cbor_offset(8) | ENDF(4)
                            // Parse CBOR to find its byte length, then the rest is payload.
                            let cbor_start = abs_cbor_offset;
                            let region = &buf[cbor_start..cbor_offset_pos];
                            match ciborium::from_reader::<ciborium::Value, _>(region) {
                                Ok(val) => {
                                    let mut cbor_bytes_buf = Vec::new();
                                    if ciborium::into_writer(&val, &mut cbor_bytes_buf).is_ok() {
                                        let cbor_len = cbor_bytes_buf.len();
                                        let payload_start = cbor_start + cbor_len;
                                        (
                                            &buf[cbor_start..cbor_start + cbor_len],
                                            &buf[payload_start..cbor_offset_pos],
                                        )
                                    } else {
                                        // Can't re-serialize; use whole region as CBOR, empty payload
                                        (region, &buf[cbor_offset_pos..cbor_offset_pos])
                                    }
                                }
                                Err(_) => {
                                    // CBOR parse fails; Level 2 will report the error.
                                    // Pass the region as CBOR with empty payload.
                                    (region, &buf[cbor_offset_pos..cbor_offset_pos])
                                }
                            }
                        };
                        data_objects.push((cbor_slice.to_vec(), payload_slice, frame_start));
                    }
                }
                obj_idx += 1;
            }
            _ => {
                // Non-data-object frames: extract payload between header and ENDF
                let payload = &buf[pos + FRAME_HEADER_SIZE..endf_offset];
                meta_frames.push((fh.frame_type, payload));
            }
        }

        // Advance past frame (with alignment padding to 8-byte boundary)
        let consumed = frame_end;
        let aligned = consumed.saturating_add(7) & !7;
        pos = if aligned <= msg_end.saturating_add(POSTAMBLE_SIZE) {
            aligned
        } else {
            consumed
        };
    }

    // Dangling preceder
    if pending_preceder {
        issues.push(err(
            ValidationLevel::Structure,
            None,
            None,
            "dangling PrecederMetadata: no DataObject frame followed".to_string(),
        ));
    }

    // Flags consistency: check each declared flag matches what we observed
    let declared = preamble.flags;
    let flag_checks: &[(u16, &str)] = &[
        (MessageFlags::HEADER_METADATA, "HEADER_METADATA"),
        (MessageFlags::FOOTER_METADATA, "FOOTER_METADATA"),
        (MessageFlags::HEADER_INDEX, "HEADER_INDEX"),
        (MessageFlags::FOOTER_INDEX, "FOOTER_INDEX"),
        (MessageFlags::HEADER_HASHES, "HEADER_HASHES"),
        (MessageFlags::FOOTER_HASHES, "FOOTER_HASHES"),
        (MessageFlags::PRECEDER_METADATA, "PRECEDER_METADATA"),
    ];
    for &(flag, name) in flag_checks {
        if declared.has(flag) != observed_flags.has(flag) {
            issues.push(warn(
                ValidationLevel::Structure,
                None,
                Some(10),
                format!("{name} flag mismatch: declared vs observed"),
            ));
        }
    }

    // Must have at least one metadata frame
    if !observed_flags.has(MessageFlags::HEADER_METADATA)
        && !observed_flags.has(MessageFlags::FOOTER_METADATA)
    {
        issues.push(err(
            ValidationLevel::Structure,
            None,
            None,
            "no metadata frame found (neither header nor footer)".to_string(),
        ));
    }

    Some(FrameWalkResult {
        meta_frames,
        data_objects,
    })
}

// ── Level 2: Metadata ───────────────────────────────────────────────────────

fn validate_metadata(
    walk: &FrameWalkResult<'_>,
    issues: &mut Vec<ValidationIssue>,
    check_canonical: bool,
) {
    // Parse the metadata frame(s)
    let mut global_meta: Option<GlobalMetadata> = None;
    let mut meta_base_len_before_normalization: Option<usize> = None;

    for (ft, payload) in &walk.meta_frames {
        match ft {
            FrameType::HeaderMetadata | FrameType::FooterMetadata => {
                if check_canonical {
                    if let Err(e) = metadata::verify_canonical_cbor(payload) {
                        issues.push(warn(
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
                        // Validate index consistency with data objects
                        let obj_count = walk.data_objects.len();
                        if idx.object_count as usize != obj_count {
                            issues.push(err(
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
                    }
                    Err(e) => {
                        issues.push(err(
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
        None => return, // Already reported as error in Level 1
    };

    // base.len() vs object count (before normalization)
    let obj_count = walk.data_objects.len();
    if let Some(base_len) = meta_base_len_before_normalization {
        if base_len > obj_count {
            issues.push(err(
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
                    ValidationLevel::Metadata,
                    Some(i),
                    None,
                    format!("object {i} descriptor CBOR is not canonical: {e}"),
                ));
            }
        }

        let desc: DataObjectDescriptor = match metadata::cbor_to_object_descriptor(cbor_bytes) {
            Ok(d) => d,
            Err(e) => {
                issues.push(err(
                    ValidationLevel::Metadata,
                    Some(i),
                    None,
                    format!("object {i} descriptor CBOR parse failed: {e}"),
                ));
                continue;
            }
        };

        // ndim / shape / strides consistency
        if desc.ndim as usize != desc.shape.len() {
            issues.push(err(
                ValidationLevel::Metadata,
                Some(i),
                None,
                format!(
                    "object {i}: ndim {} != shape.len() {}",
                    desc.ndim,
                    desc.shape.len()
                ),
            ));
        }
        if desc.strides.len() != desc.shape.len() {
            issues.push(err(
                ValidationLevel::Metadata,
                Some(i),
                None,
                format!(
                    "object {i}: strides.len() {} != shape.len() {}",
                    desc.strides.len(),
                    desc.shape.len()
                ),
            ));
        }

        // Shape product overflow
        if desc
            .shape
            .iter()
            .try_fold(1u64, |acc, &x| acc.checked_mul(x))
            .is_none()
        {
            issues.push(err(
                ValidationLevel::Metadata,
                Some(i),
                None,
                format!("object {i}: shape product overflows u64"),
            ));
        }

        // Encoding/filter/compression recognized
        if !matches!(desc.encoding.as_str(), "none" | "simple_packing") {
            issues.push(err(
                ValidationLevel::Metadata,
                Some(i),
                None,
                format!("object {i}: unknown encoding '{}'", desc.encoding),
            ));
        }
        if !matches!(desc.filter.as_str(), "none" | "shuffle") {
            issues.push(err(
                ValidationLevel::Metadata,
                Some(i),
                None,
                format!("object {i}: unknown filter '{}'", desc.filter),
            ));
        }
        let known_compressions = ["none", "szip", "zstd", "lz4", "blosc2", "zfp", "sz3"];
        if !known_compressions.contains(&desc.compression.as_str()) {
            issues.push(err(
                ValidationLevel::Metadata,
                Some(i),
                None,
                format!("object {i}: unknown compression '{}'", desc.compression),
            ));
        }

        // obj_type must not be empty
        if desc.obj_type.is_empty() {
            issues.push(err(
                ValidationLevel::Metadata,
                Some(i),
                None,
                format!("object {i}: obj_type is empty"),
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
                        ValidationLevel::Metadata,
                        Some(i),
                        None,
                        format!("object {i}: base[{i}]._reserved_ missing 'tensor' key"),
                    ));
                }
            } else {
                issues.push(err(
                    ValidationLevel::Metadata,
                    Some(i),
                    None,
                    format!("object {i}: base[{i}]._reserved_ is not a map"),
                ));
            }
        }
        // Missing _reserved_ in a base entry is only a warning —
        // the encoder auto-populates it but third-party producers may not.
    }
}

// ── Level 3: Integrity ──────────────────────────────────────────────────────

/// Check a single hash descriptor against payload, pushing issues on failure.
/// Returns true if hash was verified successfully.
fn check_hash(
    payload: &[u8],
    h: &crate::types::HashDescriptor,
    obj_idx: usize,
    issues: &mut Vec<ValidationIssue>,
) -> bool {
    match hash::verify_hash(payload, h) {
        Ok(()) => true,
        Err(TensogramError::HashMismatch { expected, actual }) => {
            issues.push(err(
                ValidationLevel::Integrity,
                Some(obj_idx),
                None,
                format!("object {obj_idx}: hash mismatch (expected {expected}, got {actual})"),
            ));
            false
        }
        Err(e) => {
            issues.push(err(
                ValidationLevel::Integrity,
                Some(obj_idx),
                None,
                format!("object {obj_idx}: hash verification error: {e}"),
            ));
            false
        }
    }
}

fn validate_integrity(walk: &FrameWalkResult<'_>, issues: &mut Vec<ValidationIssue>) -> bool {
    let mut hash_verified = false;

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
        // Parse descriptor for hash and pipeline info
        let desc: DataObjectDescriptor = match metadata::cbor_to_object_descriptor(cbor_bytes) {
            Ok(d) => d,
            Err(_) => continue, // Already reported at Level 2
        };

        // Hash verification: prefer per-object descriptor hash, fall back to hash frame
        if let Some(ref h) = desc.hash {
            if check_hash(payload, h, i, issues) {
                hash_verified = true;
            }
        } else if let Some(ref hf) = hash_frame {
            if i < hf.hashes.len() {
                let h = crate::types::HashDescriptor {
                    hash_type: hf.hash_type.clone(),
                    value: hf.hashes[i].clone(),
                };
                if check_hash(payload, &h, i, issues) {
                    hash_verified = true;
                }
            }
        } else {
            issues.push(warn(
                ValidationLevel::Integrity,
                Some(i),
                None,
                format!("object {i}: no hash available, cannot verify integrity"),
            ));
        }

        // Decompression check: try to run the decode pipeline
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
                                    ValidationLevel::Integrity,
                                    Some(i),
                                    None,
                                    format!("object {i}: decode pipeline failed: {e}"),
                                ));
                            }
                        }
                        Err(e) => {
                            issues.push(err(
                                ValidationLevel::Integrity,
                                Some(i),
                                None,
                                format!("object {i}: cannot build pipeline config: {e}"),
                            ));
                        }
                    }
                }
            }
            // Shape overflow already reported at Level 2
        }
    }

    hash_verified
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Validate a single message buffer.
///
/// Never panics — all errors become `ValidationIssue` entries.
pub fn validate_message(buf: &[u8], options: &ValidateOptions) -> ValidationReport {
    let mut issues = Vec::new();
    let mut object_count = 0;
    let mut hash_verified = false;

    let run_structure = options.mode != ValidateMode::Checksum;
    let run_metadata = matches!(
        options.mode,
        ValidateMode::Default | ValidateMode::Canonical
    );
    let run_integrity = matches!(
        options.mode,
        ValidateMode::Default | ValidateMode::Checksum | ValidateMode::Canonical
    );
    let check_canonical = options.mode == ValidateMode::Canonical;

    // Level 1: Structure
    let walk = if run_structure {
        validate_structure(buf, &mut issues)
    } else {
        // For checksum-only mode we still need to parse structure to get payloads
        validate_structure(buf, &mut Vec::new())
    };

    if let Some(ref walk) = walk {
        object_count = walk.data_objects.len();

        // Level 2: Metadata
        if run_metadata {
            validate_metadata(walk, &mut issues, check_canonical);
        }

        // Level 3: Integrity
        if run_integrity {
            hash_verified = validate_integrity(walk, &mut issues);
        }
    }

    ValidationReport {
        issues,
        object_count,
        hash_verified,
    }
}

/// Validate all messages in a `.tgm` file.
///
/// Scans the entire byte stream to detect truncated messages and
/// garbage bytes between valid messages.
pub fn validate_file(
    path: &Path,
    options: &ValidateOptions,
) -> std::io::Result<FileValidationReport> {
    let buf = std::fs::read(path)?;
    validate_buffer(&buf, options)
}

/// Validate all messages in a byte buffer (may contain multiple messages).
pub fn validate_buffer(
    buf: &[u8],
    options: &ValidateOptions,
) -> std::io::Result<FileValidationReport> {
    let mut file_issues = Vec::new();
    let mut messages = Vec::new();
    let mut pos = 0;

    while pos < buf.len() {
        // Look for TENSOGRM magic
        if pos + MAGIC.len() > buf.len() {
            // Trailing bytes too short to be a message
            if pos < buf.len() {
                file_issues.push(FileIssue {
                    byte_offset: pos,
                    length: buf.len() - pos,
                    description: format!("{} trailing bytes after last message", buf.len() - pos),
                });
            }
            break;
        }

        if &buf[pos..pos + MAGIC.len()] != MAGIC {
            // Not a message start — scan forward
            let gap_start = pos;
            pos += 1;
            while pos + MAGIC.len() <= buf.len() && &buf[pos..pos + MAGIC.len()] != MAGIC {
                pos += 1;
            }
            file_issues.push(FileIssue {
                byte_offset: gap_start,
                length: pos - gap_start,
                description: format!(
                    "{} unrecognized bytes at offset {}",
                    pos - gap_start,
                    gap_start
                ),
            });
            continue;
        }

        // Try to determine message length
        if pos + PREAMBLE_SIZE > buf.len() {
            file_issues.push(FileIssue {
                byte_offset: pos,
                length: buf.len() - pos,
                description: format!("truncated message preamble at offset {pos}"),
            });
            break;
        }

        let msg_buf = &buf[pos..];
        let msg_len = match Preamble::read_from(msg_buf) {
            Ok(preamble) if preamble.total_length > 0 => {
                let Ok(total) = usize::try_from(preamble.total_length) else {
                    file_issues.push(FileIssue {
                        byte_offset: pos,
                        length: PREAMBLE_SIZE,
                        description: format!(
                            "total_length {} overflows usize at offset {pos}",
                            preamble.total_length
                        ),
                    });
                    pos += 1;
                    continue;
                };
                if pos + total > buf.len() {
                    file_issues.push(FileIssue {
                        byte_offset: pos,
                        length: buf.len() - pos,
                        description: format!(
                            "truncated message at offset {pos}: total_length {} but only {} bytes remain",
                            total,
                            buf.len() - pos
                        ),
                    });
                    // Still validate what we have
                    buf.len() - pos
                } else {
                    total
                }
            }
            Ok(_) => {
                // Streaming mode: scan for end magic
                let mut found = None;
                let search_start = pos + PREAMBLE_SIZE;
                let mut scan_pos = search_start;
                while scan_pos + 8 <= buf.len() {
                    if &buf[scan_pos..scan_pos + 8] == wire::END_MAGIC {
                        found = Some(scan_pos + 8 - pos);
                        break;
                    }
                    scan_pos += 1;
                }
                match found {
                    Some(len) => len,
                    None => {
                        file_issues.push(FileIssue {
                            byte_offset: pos,
                            length: buf.len() - pos,
                            description: format!(
                                "streaming message at offset {pos} has no end magic"
                            ),
                        });
                        break;
                    }
                }
            }
            Err(_) => {
                // Bad preamble — already will be caught by validate_message
                // Try to find next magic to delimit
                let skip = 1;
                file_issues.push(FileIssue {
                    byte_offset: pos,
                    length: skip,
                    description: format!("invalid preamble at offset {pos}"),
                });
                pos += skip;
                continue;
            }
        };

        let msg_slice = &buf[pos..pos + msg_len];
        let report = validate_message(msg_slice, options);
        messages.push(report);
        pos += msg_len;
    }

    Ok(FileValidationReport {
        file_issues,
        messages,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{encode, EncodeOptions};
    use crate::types::{DataObjectDescriptor, GlobalMetadata};
    use crate::Dtype;
    use std::collections::BTreeMap;
    use tensogram_encodings::ByteOrder;

    fn make_test_message() -> Vec<u8> {
        let meta = GlobalMetadata::default();
        let desc = DataObjectDescriptor {
            obj_type: "ndarray".to_string(),
            ndim: 1,
            shape: vec![4],
            strides: vec![8],
            dtype: Dtype::Float64,
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        };
        let data: Vec<u8> = vec![0u8; 32]; // 4 float64 = 32 bytes
        encode(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap()
    }

    fn make_multi_object_message() -> Vec<u8> {
        let meta = GlobalMetadata::default();
        let desc = DataObjectDescriptor {
            obj_type: "ndarray".to_string(),
            ndim: 1,
            shape: vec![2],
            strides: vec![8],
            dtype: Dtype::Float64,
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        };
        let data: Vec<u8> = vec![0u8; 16];
        encode(
            &meta,
            &[(&desc, data.as_slice()), (&desc, data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap()
    }

    // ── Level 1 tests ───────────────────────────────────────────────────

    #[test]
    fn valid_message_passes_all_levels() {
        let msg = make_test_message();
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(report.is_ok(), "issues: {:?}", report.issues);
        assert_eq!(report.object_count, 1);
        assert!(report.hash_verified);
    }

    #[test]
    fn valid_multi_object_passes() {
        let msg = make_multi_object_message();
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(report.is_ok(), "issues: {:?}", report.issues);
        assert_eq!(report.object_count, 2);
    }

    #[test]
    fn empty_buffer_fails() {
        let report = validate_message(&[], &ValidateOptions::default());
        assert!(!report.is_ok());
        assert!(report.issues[0]
            .description
            .contains("too short for preamble"));
    }

    #[test]
    fn wrong_magic_fails() {
        let mut msg = make_test_message();
        msg[0..8].copy_from_slice(b"WRONGMAG");
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(!report.is_ok());
        assert!(report.issues[0].description.contains("invalid magic bytes"));
    }

    #[test]
    fn truncated_message_fails() {
        let msg = make_test_message();
        let truncated = &msg[..msg.len() / 2];
        let report = validate_message(truncated, &ValidateOptions::default());
        assert!(!report.is_ok());
    }

    #[test]
    fn bad_total_length_fails() {
        let mut msg = make_test_message();
        // Set total_length to something huge
        let bad_len: u64 = (msg.len() * 10) as u64;
        msg[16..24].copy_from_slice(&bad_len.to_be_bytes());
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(!report.is_ok());
        assert!(report.issues[0].description.contains("total_length"));
    }

    #[test]
    fn corrupted_postamble_fails() {
        let mut msg = make_test_message();
        let end = msg.len();
        // Corrupt end magic
        msg[end - 8..end].copy_from_slice(b"BADMAGIC");
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(!report.is_ok());
    }

    #[test]
    fn tiny_total_length_does_not_panic() {
        // Craft a buffer with valid magic but total_length smaller than min message size
        let mut buf = vec![0u8; 40]; // PREAMBLE_SIZE=24 + POSTAMBLE_SIZE=16
        buf[0..8].copy_from_slice(b"TENSOGRM");
        buf[8..10].copy_from_slice(&2u16.to_be_bytes()); // version 2
                                                         // Set total_length to 10 (less than PREAMBLE_SIZE + POSTAMBLE_SIZE = 40)
        buf[16..24].copy_from_slice(&10u64.to_be_bytes());
        let report = validate_message(&buf, &ValidateOptions::default());
        assert!(!report.is_ok());
        // Must not panic — should report an error about total_length being too small
        assert!(
            report.issues[0]
                .description
                .contains("smaller than minimum"),
            "got: {}",
            report.issues[0].description,
        );
    }

    // ── Level 2 tests ───────────────────────────────────────────────────

    #[test]
    fn corrupted_metadata_cbor_fails_level2() {
        let mut msg = make_test_message();
        // The metadata frame is right after the preamble (24 bytes).
        // Frame header is 16 bytes, then CBOR payload starts.
        // Corrupt some bytes in the CBOR region.
        let cbor_start = PREAMBLE_SIZE + FRAME_HEADER_SIZE;
        if cbor_start + 4 < msg.len() {
            msg[cbor_start] = 0xFF;
            msg[cbor_start + 1] = 0xFF;
            msg[cbor_start + 2] = 0xFF;
            msg[cbor_start + 3] = 0xFF;
        }
        let report = validate_message(&msg, &ValidateOptions::default());
        // Should have metadata-level errors
        let has_meta_error = report
            .issues
            .iter()
            .any(|i| i.level == ValidationLevel::Metadata);
        assert!(
            has_meta_error,
            "expected metadata error, got: {:?}",
            report.issues
        );
    }

    // ── Level 3 tests ───────────────────────────────────────────────────

    #[test]
    fn corrupted_payload_hash_mismatch() {
        let mut msg = make_test_message();
        // Find the data payload and corrupt one byte.
        // The data object frame comes after metadata + index + hash frames.
        // Corrupting a byte in the middle of the message should cause hash mismatch.
        let mid = msg.len() / 2;
        msg[mid] ^= 0xFF;
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_integrity_error = report.issues.iter().any(|i| {
            i.level == ValidationLevel::Integrity
                || i.level == ValidationLevel::Structure
                || i.level == ValidationLevel::Metadata
        });
        assert!(
            has_integrity_error || !report.is_ok(),
            "expected some error after corruption, got: {:?}",
            report.issues
        );
    }

    // ── Quick mode tests ────────────────────────────────────────────────

    #[test]
    fn quick_mode_skips_metadata_and_integrity() {
        let msg = make_test_message();
        let opts = ValidateOptions {
            mode: ValidateMode::Quick,
        };
        let report = validate_message(&msg, &opts);
        assert!(report.is_ok());
        // Should not have verified hash in quick mode
        assert!(!report.hash_verified);
    }

    // ── Checksum mode tests ─────────────────────────────────────────────

    #[test]
    fn checksum_mode_verifies_hash() {
        let msg = make_test_message();
        let opts = ValidateOptions {
            mode: ValidateMode::Checksum,
        };
        let report = validate_message(&msg, &opts);
        assert!(report.is_ok());
        assert!(report.hash_verified);
    }

    // ── File-level tests ────────────────────────────────────────────────

    #[test]
    fn validate_buffer_single_message() {
        let msg = make_test_message();
        let report = validate_buffer(&msg, &ValidateOptions::default()).unwrap();
        assert!(report.is_ok());
        assert_eq!(report.messages.len(), 1);
        assert!(report.file_issues.is_empty());
    }

    #[test]
    fn validate_buffer_two_messages() {
        let msg = make_test_message();
        let mut buf = msg.clone();
        buf.extend_from_slice(&msg);
        let report = validate_buffer(&buf, &ValidateOptions::default()).unwrap();
        assert!(report.is_ok());
        assert_eq!(report.messages.len(), 2);
    }

    #[test]
    fn validate_buffer_trailing_garbage() {
        let mut buf = make_test_message();
        buf.extend_from_slice(b"GARBAGE_TRAILING_DATA");
        let report = validate_buffer(&buf, &ValidateOptions::default()).unwrap();
        assert!(!report.file_issues.is_empty());
        assert!(report.file_issues[0]
            .description
            .contains("unrecognized bytes"));
    }

    #[test]
    fn validate_buffer_garbage_between_messages() {
        let msg = make_test_message();
        let mut buf = msg.clone();
        buf.extend_from_slice(b"GARBAGE");
        buf.extend_from_slice(&msg);
        let report = validate_buffer(&buf, &ValidateOptions::default()).unwrap();
        assert!(!report.file_issues.is_empty());
        assert_eq!(report.messages.len(), 2);
    }

    #[test]
    fn validate_buffer_truncated_second_message() {
        let msg = make_test_message();
        let mut buf = msg.clone();
        buf.extend_from_slice(&msg[..msg.len() / 2]);
        let report = validate_buffer(&buf, &ValidateOptions::default()).unwrap();
        // First message validates OK.
        // Second message is truncated: it may be parsed with errors, or
        // reported as a file-level issue.
        assert!(!report.messages.is_empty());
        let has_issue =
            !report.file_issues.is_empty() || report.messages.iter().any(|r| !r.is_ok());
        assert!(
            has_issue,
            "truncated second message should produce an issue"
        );
    }

    // ── Canonical mode test ─────────────────────────────────────────────

    #[test]
    fn canonical_mode_on_valid_message() {
        let msg = make_test_message();
        let opts = ValidateOptions {
            mode: ValidateMode::Canonical,
        };
        let report = validate_message(&msg, &opts);
        // Our encoder always produces canonical CBOR
        assert!(report.is_ok(), "issues: {:?}", report.issues);
    }
}
