// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Types for the validation API.

use serde::Serialize;

/// Validation levels, from lightest to most thorough.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ValidationLevel {
    /// Level 1: magic bytes, frame structure, lengths, ordering.
    Structure = 1,
    /// Level 2: CBOR parses, required keys present, types recognized.
    Metadata = 2,
    /// Level 3: hash verification, decompression without value interpretation.
    Integrity = 3,
    /// Level 4: full decode, NaN/Inf detection, decoded-size check.
    Fidelity = 4,
}

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum IssueSeverity {
    Error,
    Warning,
}

/// Stable machine-readable issue codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueCode {
    // ── Level 1: Structure ──
    BufferTooShort,
    InvalidMagic,
    PreambleParseFailed,
    TotalLengthOverflow,
    TotalLengthExceedsBuffer,
    TotalLengthTooSmall,
    PostambleInvalid,
    FooterOffsetOutOfRange,
    FooterOffsetMismatch,
    TruncatedFrameHeader,
    InvalidFrameHeader,
    FrameLengthOverflow,
    FrameTooSmall,
    FrameExceedsMessage,
    MissingEndMarker,
    FrameOrderViolation,
    PrecederNotFollowedByObject,
    DanglingPreceder,
    CborOffsetInvalid,
    CborBeforeBoundaryUnknown,
    DataObjectTooSmall,
    NonZeroPadding,
    FlagMismatch,
    NoMetadataFrame,

    // ── Level 2: Metadata ──
    MetadataCborParseFailed,
    MetadataCborNonCanonical,
    IndexCborParseFailed,
    IndexCountMismatch,
    IndexOffsetMismatch,
    HashFrameCborParseFailed,
    HashFrameCountMismatch,
    PrecederCborParseFailed,
    PrecederCborNonCanonical,
    PrecederBaseCountWrong,
    BaseCountExceedsObjects,
    DescriptorCborParseFailed,
    DescriptorCborNonCanonical,
    NdimShapeMismatch,
    StridesShapeMismatch,
    ShapeOverflow,
    UnknownEncoding,
    UnknownFilter,
    UnknownCompression,
    EmptyObjType,
    ReservedNotAMap,
    ReservedMissingTensor,

    // ── Level 3: Integrity ──
    HashMismatch,
    HashVerificationError,
    UnknownHashAlgorithm,
    NoHashAvailable,
    DecodePipelineFailed,
    PipelineConfigFailed,

    // ── Level 4: Fidelity ──
    DecodeObjectFailed,
    DecodedSizeMismatch,
    NanDetected,
    InfDetected,
}

/// State of decode pipeline execution for a data object.
pub(crate) enum DecodeState {
    /// Not yet attempted.
    NotDecoded,
    /// Successfully decoded.
    Decoded(Vec<u8>),
    /// Decode was attempted and failed — Level 4 should skip.
    DecodeFailed,
}

/// Per-object validation context, populated incrementally across levels.
///
/// Level 1 fills `payload` (full payload region including mask bytes
/// if the frame carries masks) and `frame_offset`.
/// Level 2 parses the descriptor and, when `descriptor.masks` is
/// `Some`, narrows `payload` to just the data-payload portion while
/// populating `mask_region` with the trailing mask bytes.  After
/// Level 2, `payload` is safe to feed to `decode_pipeline` on its
/// own.
/// Level 3 fills `decode_state` for non-raw objects.
/// Level 4 reuses decoded bytes or scans `payload` in-place for raw objects.
pub(crate) struct ObjectContext<'a> {
    /// Parsed descriptor (filled by Level 2).
    pub descriptor: Option<crate::types::DataObjectDescriptor>,
    /// True if descriptor parse was attempted and failed (prevents Level 3 retry).
    pub descriptor_failed: bool,
    /// Raw CBOR bytes for the descriptor.
    pub cbor_bytes: &'a [u8],
    /// Data-payload bytes — the pipeline-encoded portion only.
    /// Narrowed in Level 2 when `descriptor.masks` is `Some`.
    pub payload: &'a [u8],
    /// Mask-region bytes trailing the data payload.  Empty when the
    /// frame has no masks; populated in Level 2 from the descriptor's
    /// `masks` sub-map.  See `plans/WIRE_FORMAT.md` §6.5.
    pub mask_region: &'a [u8],
    /// Byte offset of the data object frame within the message.
    pub frame_offset: usize,
    /// Decode pipeline state (filled by Level 3, reused by Level 4).
    pub decode_state: DecodeState,
}

/// A single validation finding.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationIssue {
    pub code: IssueCode,
    pub level: ValidationLevel,
    pub severity: IssueSeverity,
    /// Index of the object within the message (if applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object_index: Option<usize>,
    /// Byte offset within the message buffer (if applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_offset: Option<usize>,
    pub description: String,
}

/// Options passed to `validate_message`.
///
/// Composable: `max_level` selects how deep to validate, `check_canonical`
/// adds CBOR ordering checks, `checksum_only` limits to hash verification.
#[derive(Debug, Clone)]
pub struct ValidateOptions {
    /// Highest validation level to run (levels 1..=max_level).
    pub max_level: ValidationLevel,
    /// Check RFC 8949 deterministic CBOR key ordering (opt-in).
    pub check_canonical: bool,
    /// Checksum-only mode: run Level 3 hash verification, suppress
    /// structural warnings (errors still reported).
    pub checksum_only: bool,
}

impl Default for ValidateOptions {
    fn default() -> Self {
        Self {
            max_level: ValidationLevel::Integrity,
            check_canonical: false,
            checksum_only: false,
        }
    }
}

/// Result of validating a single message.
#[derive(Debug, Clone, Serialize)]
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
#[derive(Debug, Clone, Serialize)]
pub struct FileIssue {
    pub byte_offset: usize,
    pub length: usize,
    pub description: String,
}

/// Result of validating a `.tgm` file.
#[derive(Debug, Clone, Serialize)]
pub struct FileValidationReport {
    /// Issues at the file level (gaps, trailing bytes, truncated messages).
    pub file_issues: Vec<FileIssue>,
    /// Per-message validation reports.
    pub messages: Vec<ValidationReport>,
}

impl FileValidationReport {
    /// Returns true when there are no file-level issues and all messages pass.
    ///
    /// File-level issues (gaps, trailing bytes) are treated as failures because
    /// they indicate the file is not well-formed — even though individual
    /// messages within it may be valid.
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

pub(crate) fn issue(
    code: IssueCode,
    level: ValidationLevel,
    severity: IssueSeverity,
    object_index: Option<usize>,
    byte_offset: Option<usize>,
    description: impl Into<String>,
) -> ValidationIssue {
    ValidationIssue {
        code,
        level,
        severity,
        object_index,
        byte_offset,
        description: description.into(),
    }
}

pub(crate) fn err(
    code: IssueCode,
    level: ValidationLevel,
    object_index: Option<usize>,
    byte_offset: Option<usize>,
    description: impl Into<String>,
) -> ValidationIssue {
    issue(
        code,
        level,
        IssueSeverity::Error,
        object_index,
        byte_offset,
        description,
    )
}

pub(crate) fn warn(
    code: IssueCode,
    level: ValidationLevel,
    object_index: Option<usize>,
    byte_offset: Option<usize>,
    description: impl Into<String>,
) -> ValidationIssue {
    issue(
        code,
        level,
        IssueSeverity::Warning,
        object_index,
        byte_offset,
        description,
    )
}
