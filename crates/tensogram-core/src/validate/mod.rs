//! Validation of tensogram messages and files.
//!
//! Provides `validate_message()` for checking a single message buffer and
//! `validate_file()` for checking all messages in a `.tgm` file, including
//! detection of truncated or garbage bytes between messages.

mod integrity;
mod metadata;
mod structure;
pub mod types;

use std::path::Path;

use self::integrity::validate_integrity;
use self::metadata::validate_metadata;
use self::structure::validate_structure;
pub use self::types::*;

// ── Public API ──────────────────────────────────────────────────────────────

/// Validate a single message buffer.
///
/// Never panics — all errors become `ValidationIssue` entries.
pub fn validate_message(buf: &[u8], options: &ValidateOptions) -> ValidationReport {
    let mut issues = Vec::new();
    let mut object_count = 0;
    let mut hash_verified = false;

    let report_structure = options.mode != ValidateMode::Checksum;
    let run_metadata = matches!(
        options.mode,
        ValidateMode::Default | ValidateMode::Canonical
    );
    let run_integrity = matches!(
        options.mode,
        ValidateMode::Default | ValidateMode::Checksum | ValidateMode::Canonical
    );
    let check_canonical = options.mode == ValidateMode::Canonical;

    // Level 1: Structure — always run to extract frame payloads.
    // In checksum mode, non-fatal structural warnings are suppressed,
    // but if structure parsing fails entirely (walk=None), we report
    // that as an error since we can't verify anything.
    let mut structure_issues = Vec::new();
    let walk = validate_structure(buf, &mut structure_issues);
    if walk.is_none() {
        // Structure too broken to continue — always report this
        issues.append(&mut structure_issues);
    } else if report_structure {
        issues.append(&mut structure_issues);
    }

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
/// Uses streaming I/O — only one message is in memory at a time.
/// Detects gaps and trailing bytes between messages.
pub fn validate_file(
    path: &Path,
    options: &ValidateOptions,
) -> std::io::Result<FileValidationReport> {
    use std::io::{Read, Seek, SeekFrom};

    let file_len = std::fs::metadata(path)?.len() as usize;
    let mut file = std::fs::File::open(path)?;

    let scan_result = crate::framing::scan_file(&mut file);
    let offsets = match scan_result {
        Ok(o) => o,
        Err(_) => {
            file.seek(SeekFrom::Start(0))?;
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)?;
            let report = validate_message(&buf, options);
            return Ok(FileValidationReport {
                file_issues: Vec::new(),
                messages: vec![report],
            });
        }
    };

    let mut file_issues = Vec::new();
    let mut messages = Vec::new();
    let mut expected_pos: usize = 0;

    for (offset, length) in &offsets {
        if *offset > expected_pos {
            file_issues.push(FileIssue {
                byte_offset: expected_pos,
                length: offset - expected_pos,
                description: format!(
                    "{} unrecognized bytes at offset {}",
                    offset - expected_pos,
                    expected_pos
                ),
            });
        }

        file.seek(SeekFrom::Start(*offset as u64))?;
        let mut msg_buf = vec![0u8; *length];
        file.read_exact(&mut msg_buf)?;

        let report = validate_message(&msg_buf, options);
        messages.push(report);

        expected_pos = offset + length;
    }

    if expected_pos < file_len {
        file_issues.push(FileIssue {
            byte_offset: expected_pos,
            length: file_len - expected_pos,
            description: format!(
                "{} trailing bytes after last message at offset {}",
                file_len - expected_pos,
                expected_pos
            ),
        });
    }

    Ok(FileValidationReport {
        file_issues,
        messages,
    })
}

/// Validate all messages in a byte buffer (may contain multiple messages).
///
/// For file-based validation prefer `validate_file()` which uses streaming I/O.
pub fn validate_buffer(buf: &[u8], options: &ValidateOptions) -> FileValidationReport {
    let offsets = crate::framing::scan(buf);

    let mut file_issues = Vec::new();
    let mut messages = Vec::new();
    let mut expected_pos: usize = 0;

    for (offset, length) in &offsets {
        if *offset > expected_pos {
            file_issues.push(FileIssue {
                byte_offset: expected_pos,
                length: offset - expected_pos,
                description: format!(
                    "{} unrecognized bytes at offset {}",
                    offset - expected_pos,
                    expected_pos
                ),
            });
        }

        let msg_slice = &buf[*offset..*offset + *length];
        let report = validate_message(msg_slice, options);
        messages.push(report);

        expected_pos = offset + length;
    }

    if expected_pos < buf.len() {
        file_issues.push(FileIssue {
            byte_offset: expected_pos,
            length: buf.len() - expected_pos,
            description: format!(
                "{} trailing bytes after last message at offset {}",
                buf.len() - expected_pos,
                expected_pos
            ),
        });
    }

    FileValidationReport {
        file_issues,
        messages,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{encode, EncodeOptions};
    use crate::types::{DataObjectDescriptor, GlobalMetadata};
    use crate::wire::{FRAME_HEADER_SIZE, PREAMBLE_SIZE};
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
        let data: Vec<u8> = vec![0u8; 32];
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
        assert_eq!(report.issues[0].code, IssueCode::BufferTooShort);
    }

    #[test]
    fn wrong_magic_fails() {
        let mut msg = make_test_message();
        msg[0..8].copy_from_slice(b"WRONGMAG");
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(!report.is_ok());
        assert_eq!(report.issues[0].code, IssueCode::InvalidMagic);
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
        let bad_len: u64 = (msg.len() * 10) as u64;
        msg[16..24].copy_from_slice(&bad_len.to_be_bytes());
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(!report.is_ok());
        assert_eq!(report.issues[0].code, IssueCode::TotalLengthExceedsBuffer);
    }

    #[test]
    fn corrupted_postamble_fails() {
        let mut msg = make_test_message();
        let end = msg.len();
        msg[end - 8..end].copy_from_slice(b"BADMAGIC");
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(!report.is_ok());
    }

    #[test]
    fn tiny_total_length_does_not_panic() {
        let mut buf = vec![0u8; 40];
        buf[0..8].copy_from_slice(b"TENSOGRM");
        buf[8..10].copy_from_slice(&2u16.to_be_bytes());
        buf[16..24].copy_from_slice(&10u64.to_be_bytes());
        let report = validate_message(&buf, &ValidateOptions::default());
        assert!(!report.is_ok());
        assert_eq!(report.issues[0].code, IssueCode::TotalLengthTooSmall);
    }

    // ── Level 2 tests ───────────────────────────────────────────────────

    #[test]
    fn corrupted_metadata_cbor_fails_level2() {
        let mut msg = make_test_message();
        let cbor_start = PREAMBLE_SIZE + FRAME_HEADER_SIZE;
        if cbor_start + 4 < msg.len() {
            msg[cbor_start] = 0xFF;
            msg[cbor_start + 1] = 0xFF;
            msg[cbor_start + 2] = 0xFF;
            msg[cbor_start + 3] = 0xFF;
        }
        let report = validate_message(&msg, &ValidateOptions::default());
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
        use crate::wire::POSTAMBLE_SIZE;
        let mut msg = make_test_message();
        // Corrupt a byte inside the last data object frame, just before
        // the footer frames (well inside the message, avoiding padding).
        // The postamble is the last 16 bytes; footer frames precede it.
        // Corrupt a byte near the middle of the data region.
        let target = PREAMBLE_SIZE + FRAME_HEADER_SIZE + 20;
        if target < msg.len() - POSTAMBLE_SIZE {
            msg[target] ^= 0xFF;
        }
        let report = validate_message(&msg, &ValidateOptions::default());
        // Should have some error (metadata parse, hash mismatch, or structure)
        assert!(
            !report.is_ok(),
            "expected error after corruption, got: {:?}",
            report.issues
        );
    }

    // ── Mode tests ──────────────────────────────────────────────────────

    #[test]
    fn quick_mode_skips_metadata_and_integrity() {
        let msg = make_test_message();
        let opts = ValidateOptions {
            mode: ValidateMode::Quick,
        };
        let report = validate_message(&msg, &opts);
        assert!(report.is_ok());
        assert!(!report.hash_verified);
    }

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

    #[test]
    fn checksum_mode_on_broken_message_fails() {
        let mut msg = make_test_message();
        let end = msg.len();
        msg[end - 8..end].copy_from_slice(b"BADMAGIC");
        let opts = ValidateOptions {
            mode: ValidateMode::Checksum,
        };
        let report = validate_message(&msg, &opts);
        assert!(
            !report.is_ok(),
            "broken message should fail even in checksum mode"
        );
    }

    #[test]
    fn canonical_mode_on_valid_message() {
        let msg = make_test_message();
        let opts = ValidateOptions {
            mode: ValidateMode::Canonical,
        };
        let report = validate_message(&msg, &opts);
        assert!(report.is_ok(), "issues: {:?}", report.issues);
    }

    // ── File-level tests ────────────────────────────────────────────────

    #[test]
    fn validate_buffer_single_message() {
        let msg = make_test_message();
        let report = validate_buffer(&msg, &ValidateOptions::default());
        assert!(report.is_ok());
        assert_eq!(report.messages.len(), 1);
        assert!(report.file_issues.is_empty());
    }

    #[test]
    fn validate_buffer_two_messages() {
        let msg = make_test_message();
        let mut buf = msg.clone();
        buf.extend_from_slice(&msg);
        let report = validate_buffer(&buf, &ValidateOptions::default());
        assert!(report.is_ok());
        assert_eq!(report.messages.len(), 2);
    }

    #[test]
    fn validate_buffer_trailing_garbage() {
        let mut buf = make_test_message();
        buf.extend_from_slice(b"GARBAGE_TRAILING_DATA");
        let report = validate_buffer(&buf, &ValidateOptions::default());
        assert!(!report.file_issues.is_empty());
    }

    #[test]
    fn validate_buffer_garbage_between_messages() {
        let msg = make_test_message();
        let mut buf = msg.clone();
        buf.extend_from_slice(b"GARBAGE");
        buf.extend_from_slice(&msg);
        let report = validate_buffer(&buf, &ValidateOptions::default());
        assert!(!report.file_issues.is_empty());
        assert_eq!(report.messages.len(), 2);
    }

    #[test]
    fn validate_buffer_truncated_second_message() {
        let msg = make_test_message();
        let mut buf = msg.clone();
        buf.extend_from_slice(&msg[..msg.len() / 2]);
        let report = validate_buffer(&buf, &ValidateOptions::default());
        assert!(!report.messages.is_empty());
        let has_issue =
            !report.file_issues.is_empty() || report.messages.iter().any(|r| !r.is_ok());
        assert!(
            has_issue,
            "truncated second message should produce an issue"
        );
    }

    // ── Streaming message tests ─────────────────────────────────────────

    #[test]
    fn streaming_message_validates() {
        use crate::streaming::StreamingEncoder;

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
        let data = vec![0u8; 32];

        let mut buf = Vec::new();
        let mut enc = StreamingEncoder::new(&mut buf, &meta, &EncodeOptions::default()).unwrap();
        enc.write_object(&desc, &data).unwrap();
        enc.finish().unwrap();

        let report = validate_message(&buf, &ValidateOptions::default());
        assert!(
            report.is_ok(),
            "streaming message should validate: {:?}",
            report.issues
        );
        assert_eq!(report.object_count, 1);
        assert!(report.hash_verified);
    }

    #[test]
    fn streaming_message_footer_metadata_validates() {
        use crate::streaming::StreamingEncoder;

        let meta = GlobalMetadata::default();
        let desc = DataObjectDescriptor {
            obj_type: "ndarray".to_string(),
            ndim: 1,
            shape: vec![2],
            strides: vec![4],
            dtype: Dtype::Float32,
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        };
        let data = vec![0u8; 8];

        let mut buf = Vec::new();
        let mut enc = StreamingEncoder::new(&mut buf, &meta, &EncodeOptions::default()).unwrap();
        enc.write_object(&desc, &data).unwrap();
        enc.write_object(&desc, &data).unwrap();
        enc.finish().unwrap();

        let report = validate_message(&buf, &ValidateOptions::default());
        assert!(report.is_ok(), "issues: {:?}", report.issues);
        assert_eq!(report.object_count, 2);
    }

    // ── Hash tests ──────────────────────────────────────────────────────

    #[test]
    fn hash_verified_requires_all_objects() {
        let msg = make_test_message();
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(report.hash_verified);
    }

    #[test]
    fn hash_not_verified_without_hash() {
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
        let data = vec![0u8; 32];
        let opts = EncodeOptions {
            hash_algorithm: None,
            ..EncodeOptions::default()
        };
        let msg = encode(&meta, &[(&desc, data.as_slice())], &opts).unwrap();
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(!report.hash_verified);
    }

    // ── Issue code tests ────────────────────────────────────────────────

    #[test]
    fn issue_codes_are_stable_strings() {
        // Verify serde serialization of issue codes
        let code = IssueCode::HashMismatch;
        let json = serde_json::to_string(&code).unwrap();
        assert_eq!(json, r#""hash_mismatch""#);
    }

    #[test]
    fn report_serializes_to_json() {
        let msg = make_test_message();
        let report = validate_message(&msg, &ValidateOptions::default());
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"object_count\":1"));
        assert!(json.contains("\"hash_verified\":true"));
    }
}
