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
