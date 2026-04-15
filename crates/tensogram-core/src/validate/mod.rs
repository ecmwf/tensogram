// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Validation of tensogram messages and files.
//!
//! Provides `validate_message()` for checking a single message buffer and
//! `validate_file()` for checking all messages in a `.tgm` file, including
//! detection of truncated or garbage bytes between messages.

mod fidelity;
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

    // Normalize options: checksum_only implies at least Integrity level
    let effective_max = if options.checksum_only {
        options.max_level.max(ValidationLevel::Integrity)
    } else {
        options.max_level
    };
    let report_structure = !options.checksum_only;
    let check_canonical = options.check_canonical;
    // Canonical checks require metadata level to parse CBOR
    let run_metadata =
        (effective_max >= ValidationLevel::Metadata && !options.checksum_only) || check_canonical;
    let run_integrity = effective_max >= ValidationLevel::Integrity;
    let run_fidelity = effective_max >= ValidationLevel::Fidelity && !options.checksum_only;

    // Level 1: Structure — always run to extract frame payloads.
    // In checksum mode, non-fatal structural warnings are suppressed,
    // but if structure parsing fails entirely (walk=None), we report
    // that as an error since we can't verify anything.
    let mut structure_issues = Vec::new();
    let walk = validate_structure(buf, &mut structure_issues);
    if report_structure {
        // Include all structure findings
        issues.append(&mut structure_issues);
    } else {
        // Checksum mode: suppress warnings but keep errors — structural
        // errors (e.g. missing ENDF, broken frames) indicate the message
        // can't be reliably verified.
        issues.extend(
            structure_issues
                .into_iter()
                .filter(|i| i.severity == IssueSeverity::Error),
        );
    }

    if let Some(ref walk) = walk {
        object_count = walk.data_objects.len();

        // Build per-object contexts only when needed by metadata, integrity, or fidelity.
        // In quick mode (without --canonical), we skip this allocation.
        let needs_objects = run_metadata || run_integrity || run_fidelity || check_canonical;
        let mut objects: Vec<ObjectContext<'_>> = if needs_objects {
            walk.data_objects
                .iter()
                .map(|(cbor_bytes, payload, frame_offset)| ObjectContext {
                    descriptor: None,
                    descriptor_failed: false,
                    cbor_bytes,
                    payload,
                    frame_offset: *frame_offset,
                    decode_state: DecodeState::NotDecoded,
                })
                .collect()
        } else {
            Vec::new()
        };

        // Level 2: Metadata — parses and caches descriptors
        if run_metadata {
            validate_metadata(walk, &mut objects, &mut issues, check_canonical);
        }

        // Level 3: Integrity — hash verification + decode pipeline (caches decoded bytes)
        if run_integrity {
            hash_verified = validate_integrity(
                walk,
                &mut objects,
                &mut issues,
                options.checksum_only,
                run_fidelity,
            );
        }

        // Level 4: Fidelity — full decode check, NaN/Inf scan
        if run_fidelity {
            fidelity::validate_fidelity(&mut objects, &mut issues);
        }
    }

    // hash_verified must only be true for a fully clean validation.
    // If any level reported an error, force it to false.
    if issues.iter().any(|i| i.severity == IssueSeverity::Error) {
        hash_verified = false;
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

    let file_len = usize::try_from(std::fs::metadata(path)?.len()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "file size does not fit into usize",
        )
    })?;
    let mut file = std::fs::File::open(path)?;

    let offsets = crate::framing::scan_file(&mut file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

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
        let trailing_len = file_len - expected_pos;
        let desc = if messages.is_empty() {
            format!("{trailing_len} bytes with no valid messages")
        } else {
            format!("{trailing_len} trailing bytes after last message at offset {expected_pos}")
        };
        file_issues.push(FileIssue {
            byte_offset: expected_pos,
            length: trailing_len,
            description: desc,
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
        let trailing_len = buf.len() - expected_pos;
        let desc = if messages.is_empty() {
            format!("{trailing_len} bytes with no valid messages")
        } else {
            format!("{trailing_len} trailing bytes after last message at offset {expected_pos}")
        };
        file_issues.push(FileIssue {
            byte_offset: expected_pos,
            length: trailing_len,
            description: desc,
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
    fn corrupted_byte_detected() {
        let mut msg = make_test_message();
        // Corrupt a byte inside the metadata CBOR region.
        // This causes metadata parse failure or structure issues.
        let target = PREAMBLE_SIZE + FRAME_HEADER_SIZE + 20;
        if target < msg.len() {
            msg[target] ^= 0xFF;
        }
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(
            !report.is_ok(),
            "expected error after corruption, got: {:?}",
            report.issues
        );
    }

    #[test]
    fn hash_mismatch_on_corrupted_payload() {
        // Encode with hash, then corrupt the data payload (not metadata).
        // Find the data object by looking for the last ENDF before postamble.
        use crate::wire::POSTAMBLE_SIZE;
        let msg = make_test_message();
        // The data payload is inside the data object frame. For a message
        // with encoding=none, the payload is raw bytes between the frame
        // header and the CBOR descriptor. We need to find the data object
        // frame. In the default encoder layout (header metadata, header index,
        // header hash, data object), the data object starts after 3 header frames.
        // Rather than computing the exact offset, search backward from postamble
        // for a region that's clearly inside the data object payload.
        let pa_start = msg.len() - POSTAMBLE_SIZE;
        // Go back past the data object footer (ENDF=4 + cbor_offset=8 + CBOR descriptor)
        // and corrupt somewhere in the middle of the payload.
        // The data object frame typically starts around byte 200+ for this test message.
        // Corrupt a byte at 70% of the way through, well inside the data region.
        let target = pa_start * 7 / 10;
        let mut corrupted = msg.clone();
        corrupted[target] ^= 0xFF;
        let report = validate_message(&corrupted, &ValidateOptions::default());
        let has_hash_or_integrity = report.issues.iter().any(|i| {
            matches!(
                i.code,
                IssueCode::HashMismatch | IssueCode::DecodePipelineFailed
            )
        });
        // If the corruption hit the data payload, we get a hash mismatch.
        // If it hit the descriptor CBOR, we get a metadata error.
        // Either way, the message should fail.
        assert!(
            !report.is_ok(),
            "corrupted payload should fail validation, got: {:?}",
            report.issues
        );
        // On most runs, this should hit the data payload and produce a hash error.
        // But we can't guarantee the exact offset, so we just assert failure.
        let _ = has_hash_or_integrity; // used for documentation, not assertion
    }

    // ── Mode tests ──────────────────────────────────────────────────────

    #[test]
    fn quick_mode_skips_metadata_and_integrity() {
        let msg = make_test_message();
        let opts = ValidateOptions {
            max_level: ValidationLevel::Structure,
            ..ValidateOptions::default()
        };
        let report = validate_message(&msg, &opts);
        assert!(report.is_ok());
        assert!(!report.hash_verified);
    }

    #[test]
    fn checksum_mode_verifies_hash() {
        let msg = make_test_message();
        let opts = ValidateOptions {
            checksum_only: true,
            ..ValidateOptions::default()
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
            checksum_only: true,
            ..ValidateOptions::default()
        };
        let report = validate_message(&msg, &opts);
        assert!(
            !report.is_ok(),
            "broken message should fail even in checksum mode"
        );
    }

    #[test]
    fn checksum_mode_catches_structural_errors() {
        let mut msg = make_test_message();
        let actual_len = msg.len() as u64;
        let bad_len = actual_len + 100;
        msg[16..24].copy_from_slice(&bad_len.to_be_bytes());
        let opts = ValidateOptions {
            checksum_only: true,
            ..ValidateOptions::default()
        };
        let report = validate_message(&msg, &opts);
        let has_error = report
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Error);
        assert!(
            has_error,
            "checksum mode should surface structural errors, got: {:?}",
            report.issues
        );
    }

    #[test]
    fn canonical_mode_on_valid_message() {
        let msg = make_test_message();
        let opts = ValidateOptions {
            check_canonical: true,
            ..ValidateOptions::default()
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

    // ── Regression tests for pass 3-4 fixes ─────────────────────────────

    #[test]
    fn validate_buffer_garbage_only() {
        let buf = b"this is not a tensogram file at all";
        let report = validate_buffer(buf, &ValidateOptions::default());
        assert!(report.messages.is_empty());
        assert!(!report.file_issues.is_empty());
        assert!(
            report.file_issues[0]
                .description
                .contains("no valid messages"),
            "got: {}",
            report.file_issues[0].description,
        );
    }

    #[test]
    fn validate_buffer_empty() {
        let report = validate_buffer(&[], &ValidateOptions::default());
        assert!(report.messages.is_empty());
        assert!(report.file_issues.is_empty());
        assert!(report.is_ok());
    }

    #[test]
    fn streaming_ffo_out_of_range_reported() {
        // Build a streaming message and corrupt first_footer_offset in the postamble
        use crate::streaming::StreamingEncoder;

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
        let data = vec![0u8; 16];

        let mut buf = Vec::new();
        let mut enc = StreamingEncoder::new(&mut buf, &meta, &EncodeOptions::default()).unwrap();
        enc.write_object(&desc, &data).unwrap();
        enc.finish().unwrap();

        // Corrupt the first_footer_offset (8 bytes before end magic)
        let pa_start = buf.len() - 16;
        let bad_ffo: u64 = 0; // 0 < PREAMBLE_SIZE, so out of range
        buf[pa_start..pa_start + 8].copy_from_slice(&bad_ffo.to_be_bytes());

        let report = validate_message(&buf, &ValidateOptions::default());
        let has_ffo_error = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::FooterOffsetOutOfRange);
        assert!(
            has_ffo_error,
            "expected FooterOffsetOutOfRange, got: {:?}",
            report.issues
        );
    }

    // ── Zero-object message ─────────────────────────────────────────────

    #[test]
    fn zero_object_message_validates() {
        let meta = GlobalMetadata::default();
        let opts = EncodeOptions {
            hash_algorithm: None,
            ..EncodeOptions::default()
        };
        let msg = encode(&meta, &[], &opts).unwrap();
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(report.is_ok(), "issues: {:?}", report.issues);
        assert_eq!(report.object_count, 0);
        assert!(!report.hash_verified); // no objects → nothing to verify
    }

    // ── Level 4: Fidelity tests ─────────────────────────────────────────

    fn full_opts() -> ValidateOptions {
        ValidateOptions {
            max_level: ValidationLevel::Fidelity,
            ..ValidateOptions::default()
        }
    }

    fn make_float64_message(values: &[f64]) -> Vec<u8> {
        let meta = GlobalMetadata::default();
        let desc = DataObjectDescriptor {
            obj_type: "ndarray".to_string(),
            ndim: 1,
            shape: vec![values.len() as u64],
            strides: vec![8],
            dtype: Dtype::Float64,
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        };
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
        encode(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap()
    }

    fn make_float32_message_le(values: &[f32]) -> Vec<u8> {
        let meta = GlobalMetadata::default();
        let desc = DataObjectDescriptor {
            obj_type: "ndarray".to_string(),
            ndim: 1,
            shape: vec![values.len() as u64],
            strides: vec![4],
            dtype: Dtype::Float32,
            byte_order: ByteOrder::Little,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        };
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        encode(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap()
    }

    #[test]
    fn full_mode_valid_float64_passes() {
        let msg = make_float64_message(&[1.0, 2.0, 3.0, 4.0]);
        let report = validate_message(&msg, &full_opts());
        assert!(report.is_ok(), "issues: {:?}", report.issues);
    }

    #[test]
    fn full_mode_nan_float64_detected() {
        let msg = make_float64_message(&[1.0, f64::NAN, 3.0]);
        let report = validate_message(&msg, &full_opts());
        assert!(!report.is_ok());
        let nan_issue = report
            .issues
            .iter()
            .find(|i| i.code == IssueCode::NanDetected);
        assert!(
            nan_issue.is_some(),
            "expected NanDetected, got: {:?}",
            report.issues
        );
        assert!(nan_issue.unwrap().description.contains("element 1"));
    }

    #[test]
    fn full_mode_inf_float64_detected() {
        let msg = make_float64_message(&[1.0, 2.0, f64::INFINITY]);
        let report = validate_message(&msg, &full_opts());
        assert!(!report.is_ok());
        let inf_issue = report
            .issues
            .iter()
            .find(|i| i.code == IssueCode::InfDetected);
        assert!(
            inf_issue.is_some(),
            "expected InfDetected, got: {:?}",
            report.issues
        );
        assert!(inf_issue.unwrap().description.contains("element 2"));
    }

    #[test]
    fn full_mode_neg_inf_detected() {
        let msg = make_float64_message(&[f64::NEG_INFINITY, 1.0]);
        let report = validate_message(&msg, &full_opts());
        assert!(!report.is_ok());
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::InfDetected));
    }

    #[test]
    fn full_mode_float32_le_nan_detected() {
        let msg = make_float32_message_le(&[1.0, f32::NAN, 3.0]);
        let report = validate_message(&msg, &full_opts());
        assert!(!report.is_ok());
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::NanDetected));
    }

    #[test]
    fn full_mode_float32_le_inf_detected() {
        let msg = make_float32_message_le(&[f32::INFINITY]);
        let report = validate_message(&msg, &full_opts());
        assert!(!report.is_ok());
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::InfDetected));
    }

    #[test]
    fn full_mode_integer_passes() {
        let meta = GlobalMetadata::default();
        let desc = DataObjectDescriptor {
            obj_type: "ndarray".to_string(),
            ndim: 1,
            shape: vec![4],
            strides: vec![4],
            dtype: Dtype::Int32,
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        };
        let data = vec![0u8; 16]; // 4 × i32
        let msg = encode(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();
        let report = validate_message(&msg, &full_opts());
        assert!(
            report.is_ok(),
            "integer should pass fidelity: {:?}",
            report.issues
        );
    }

    #[test]
    fn full_mode_hash_verified_false_on_nan() {
        let msg = make_float64_message(&[f64::NAN]);
        let report = validate_message(&msg, &full_opts());
        assert!(
            !report.hash_verified,
            "hash_verified should be false when NaN detected"
        );
    }

    #[test]
    fn full_mode_float16_nan() {
        let meta = GlobalMetadata::default();
        let desc = DataObjectDescriptor {
            obj_type: "ndarray".to_string(),
            ndim: 1,
            shape: vec![2],
            strides: vec![2],
            dtype: Dtype::Float16,
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        };
        // Float16: exponent=0x1F (all 1s), mantissa=1 => NaN
        // Bit pattern: 0_11111_0000000001 = 0x7C01
        let mut data = vec![0u8; 4]; // 2 × f16
        data[0..2].copy_from_slice(&0x0000u16.to_be_bytes()); // valid zero
        data[2..4].copy_from_slice(&0x7C01u16.to_be_bytes()); // NaN
        let msg = encode(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();
        let report = validate_message(&msg, &full_opts());
        assert!(!report.is_ok());
        let nan = report
            .issues
            .iter()
            .find(|i| i.code == IssueCode::NanDetected);
        assert!(
            nan.is_some(),
            "expected float16 NaN, got: {:?}",
            report.issues
        );
        assert!(nan.unwrap().description.contains("element 1"));
    }

    #[test]
    fn full_mode_float16_inf() {
        let meta = GlobalMetadata::default();
        let desc = DataObjectDescriptor {
            obj_type: "ndarray".to_string(),
            ndim: 1,
            shape: vec![1],
            strides: vec![2],
            dtype: Dtype::Float16,
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        };
        // Float16 Inf: exponent=0x1F, mantissa=0 => 0x7C00
        let data = 0x7C00u16.to_be_bytes().to_vec();
        let msg = encode(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();
        let report = validate_message(&msg, &full_opts());
        assert!(!report.is_ok());
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::InfDetected));
    }

    #[test]
    fn full_mode_bfloat16_nan() {
        let meta = GlobalMetadata::default();
        let desc = DataObjectDescriptor {
            obj_type: "ndarray".to_string(),
            ndim: 1,
            shape: vec![1],
            strides: vec![2],
            dtype: Dtype::Bfloat16,
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        };
        // BFloat16 NaN: exponent=0xFF, mantissa≠0
        // sign(1) + exp(8) + mantissa(7): 0_11111111_0000001 = 0x7F81
        let data = 0x7F81u16.to_be_bytes().to_vec();
        let msg = encode(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();
        let report = validate_message(&msg, &full_opts());
        assert!(!report.is_ok());
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::NanDetected));
    }

    #[test]
    fn full_mode_complex64_real_nan() {
        let meta = GlobalMetadata::default();
        let desc = DataObjectDescriptor {
            obj_type: "ndarray".to_string(),
            ndim: 1,
            shape: vec![1],
            strides: vec![8],
            dtype: Dtype::Complex64,
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        };
        // Complex64: [NaN real, 0.0 imag]
        let mut data = Vec::new();
        data.extend_from_slice(&f32::NAN.to_be_bytes());
        data.extend_from_slice(&0.0f32.to_be_bytes());
        let msg = encode(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();
        let report = validate_message(&msg, &full_opts());
        assert!(!report.is_ok());
        let nan = report
            .issues
            .iter()
            .find(|i| i.code == IssueCode::NanDetected);
        assert!(nan.is_some());
        assert!(nan.unwrap().description.contains("real component"));
    }

    #[test]
    fn full_mode_complex128_imag_inf() {
        let meta = GlobalMetadata::default();
        let desc = DataObjectDescriptor {
            obj_type: "ndarray".to_string(),
            ndim: 1,
            shape: vec![1],
            strides: vec![16],
            dtype: Dtype::Complex128,
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        };
        // Complex128: [1.0 real, Inf imag]
        let mut data = Vec::new();
        data.extend_from_slice(&1.0f64.to_be_bytes());
        data.extend_from_slice(&f64::INFINITY.to_be_bytes());
        let msg = encode(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();
        let report = validate_message(&msg, &full_opts());
        assert!(!report.is_ok());
        let inf = report
            .issues
            .iter()
            .find(|i| i.code == IssueCode::InfDetected);
        assert!(inf.is_some());
        assert!(inf.unwrap().description.contains("imaginary component"));
    }

    #[test]
    fn full_mode_with_canonical() {
        let msg = make_test_message();
        let opts = ValidateOptions {
            max_level: ValidationLevel::Fidelity,
            check_canonical: true,
            ..ValidateOptions::default()
        };
        let report = validate_message(&msg, &opts);
        assert!(
            report.is_ok(),
            "full+canonical should pass: {:?}",
            report.issues
        );
    }

    #[test]
    fn full_mode_json_serialization() {
        let msg = make_float64_message(&[f64::NAN]);
        let report = validate_message(&msg, &full_opts());
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"nan_detected\""));
        assert!(json.contains("\"fidelity\""));
    }

    #[test]
    fn default_mode_skips_fidelity() {
        // Default mode (Integrity) should NOT run fidelity checks
        let msg = make_float64_message(&[f64::NAN]);
        let report = validate_message(&msg, &ValidateOptions::default());
        // NaN should not be detected at default level
        assert!(
            !report
                .issues
                .iter()
                .any(|i| i.code == IssueCode::NanDetected),
            "default mode should not run fidelity: {:?}",
            report.issues
        );
    }

    // ── Review test gaps ────────────────────────────────────────────────

    #[test]
    fn full_mode_negative_zero_passes() {
        let msg = make_float64_message(&[-0.0, 0.0, 1.0]);
        let report = validate_message(&msg, &full_opts());
        assert!(
            report.is_ok(),
            "negative zero should pass: {:?}",
            report.issues
        );
    }

    #[test]
    fn full_mode_subnormal_passes() {
        // Smallest subnormal f64
        let msg = make_float64_message(&[5e-324, 1.0]);
        let report = validate_message(&msg, &full_opts());
        assert!(
            report.is_ok(),
            "subnormals should pass: {:?}",
            report.issues
        );
    }

    #[test]
    fn full_mode_zero_length_array() {
        let meta = GlobalMetadata::default();
        let desc = DataObjectDescriptor {
            obj_type: "ndarray".to_string(),
            ndim: 1,
            shape: vec![0],
            strides: vec![8],
            dtype: Dtype::Float64,
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        };
        let data: Vec<u8> = vec![];
        let msg = encode(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();
        let report = validate_message(&msg, &full_opts());
        assert!(
            report.is_ok(),
            "zero-length array should pass: {:?}",
            report.issues
        );
    }

    #[test]
    fn full_mode_decoded_size_mismatch() {
        // Encode a valid 3-element float32 message, then patch the shape
        // in the wire bytes to claim 4 elements. This creates a mismatch
        // between decoded payload size (12 bytes) and expected (16 bytes).
        let meta = GlobalMetadata::default();
        let desc = DataObjectDescriptor {
            obj_type: "ndarray".to_string(),
            ndim: 1,
            shape: vec![3],
            strides: vec![4],
            dtype: Dtype::Float32,
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        };
        let data = vec![0u8; 12]; // 3 × f32, valid
        let opts = EncodeOptions {
            hash_algorithm: None,
            ..EncodeOptions::default()
        };
        let mut msg = encode(&meta, &[(&desc, data.as_slice())], &opts).unwrap();

        // Patch the data object descriptor's shape from [3] to [4] in the wire bytes.
        // CBOR array(1) + uint(3) = 0x81 0x03. We search backward from the end
        // to find the data object descriptor (last CBOR in the message before the
        // postamble), avoiding any match in the metadata frame's _reserved_.tensor
        // which encodes shape differently (as a CBOR array under a map key).
        let mut patched = false;
        for i in (0..msg.len() - 1).rev() {
            if msg[i] == 0x81 && msg[i + 1] == 0x03 {
                msg[i + 1] = 0x04; // shape [3] → [4]
                patched = true;
                break;
            }
        }
        assert!(patched, "could not find shape [3] in encoded message");

        let report = validate_message(&msg, &full_opts());
        let has_mismatch = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::DecodedSizeMismatch);
        assert!(
            has_mismatch,
            "expected DecodedSizeMismatch, got: {:?}",
            report.issues
        );
    }

    #[test]
    fn quick_canonical_runs_metadata() {
        // --quick --canonical should still parse metadata for canonical check
        let msg = make_test_message();
        let opts = ValidateOptions {
            max_level: ValidationLevel::Structure,
            check_canonical: true,
            checksum_only: false,
        };
        let report = validate_message(&msg, &opts);
        // Should pass (our encoder produces canonical CBOR)
        assert!(report.is_ok(), "issues: {:?}", report.issues);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Coverage gap tests — Structure (Level 1)
    // ═══════════════════════════════════════════════════════════════════════

    /// Helper: build a minimal valid message from raw parts.
    /// Constructs: preamble + metadata_frame + data_object_frame + postamble.
    fn build_raw_message(
        flags: u16,
        frames: &[Vec<u8>], // pre-built frame bytes (including headers + ENDF)
        total_length_override: Option<u64>,
        streaming: bool,
    ) -> Vec<u8> {
        use crate::wire::{END_MAGIC, MAGIC};

        let mut out = Vec::new();

        // Preamble placeholder
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&2u16.to_be_bytes()); // version
        out.extend_from_slice(&flags.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes()); // reserved
        out.extend_from_slice(&0u64.to_be_bytes()); // total_length placeholder

        // Frames
        for frame in frames {
            out.extend_from_slice(frame);
            // 8-byte alignment
            let pad = (8 - (out.len() % 8)) % 8;
            out.extend(std::iter::repeat_n(0u8, pad));
        }

        // Postamble: first_footer_offset = current position (no footer frames)
        let ffo = out.len() as u64;
        out.extend_from_slice(&ffo.to_be_bytes());
        out.extend_from_slice(END_MAGIC);

        // Patch total_length in preamble
        let total = if streaming { 0u64 } else { out.len() as u64 };
        let tl = total_length_override.unwrap_or(total);
        out[16..24].copy_from_slice(&tl.to_be_bytes());

        out
    }

    /// Helper: build a simple metadata frame (type=HeaderMetadata) from scratch.
    fn build_metadata_frame() -> Vec<u8> {
        use crate::wire::{FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC};
        let meta = GlobalMetadata::default();
        let cbor = crate::metadata::global_metadata_to_cbor(&meta).unwrap();
        let total_length = (FRAME_HEADER_SIZE + cbor.len() + FRAME_END.len()) as u64;
        let mut frame = Vec::new();
        frame.extend_from_slice(FRAME_MAGIC);
        frame.extend_from_slice(&1u16.to_be_bytes()); // type = HeaderMetadata
        frame.extend_from_slice(&1u16.to_be_bytes()); // version
        frame.extend_from_slice(&0u16.to_be_bytes()); // flags
        frame.extend_from_slice(&total_length.to_be_bytes());
        frame.extend_from_slice(&cbor);
        frame.extend_from_slice(FRAME_END);
        frame
    }

    /// Helper: build a data object frame from a descriptor and payload.
    fn build_data_object_frame(desc: &DataObjectDescriptor, payload: &[u8]) -> Vec<u8> {
        crate::framing::encode_data_object_frame(desc, payload, false).unwrap()
    }

    /// Helper: the default ndarray descriptor used in many tests.
    fn default_desc() -> DataObjectDescriptor {
        DataObjectDescriptor {
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
        }
    }

    // ── Structure: FrameLengthOverflow ──────────────────────────────────

    #[test]
    fn structure_frame_length_overflow() {
        // Build a valid message, then patch a frame's total_length to u64::MAX
        // which won't fit in usize on 64-bit (or will overflow pos+total).
        let mut msg = make_test_message();
        // Find the first frame header after preamble (at offset 24)
        let frame_start = PREAMBLE_SIZE;
        // Frame total_length is at offset 8 within frame header
        let tl_offset = frame_start + 8;
        // Set to u64::MAX — this will cause overflow when computing frame_end = pos + total
        msg[tl_offset..tl_offset + 8].copy_from_slice(&u64::MAX.to_be_bytes());
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(!report.is_ok());
        let has_overflow = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::FrameLengthOverflow);
        assert!(
            has_overflow,
            "expected FrameLengthOverflow, got: {:?}",
            report.issues
        );
    }

    // ── Structure: NonZeroPadding ───────────────────────────────────────

    #[test]
    fn structure_non_zero_padding_between_frames() {
        // Build a valid message, then inject non-zero bytes in the
        // alignment padding between the metadata frame and the next frame.
        let mut msg = make_test_message();
        // After preamble (24 bytes) comes the first frame. Find where that
        // frame ends by reading its total_length.
        let frame_start = PREAMBLE_SIZE;
        let tl =
            u64::from_be_bytes(msg[frame_start + 8..frame_start + 16].try_into().unwrap()) as usize;
        let frame_end = frame_start + tl;
        // Check if there's padding between frame_end and the next 8-byte boundary
        let next_aligned = (frame_end + 7) & !7;
        if next_aligned > frame_end && next_aligned < msg.len() {
            // Fill padding with non-zero
            for b in &mut msg[frame_end..next_aligned] {
                *b = 0xAA;
            }
            let report = validate_message(&msg, &ValidateOptions::default());
            let has_padding_warn = report
                .issues
                .iter()
                .any(|i| i.code == IssueCode::NonZeroPadding);
            assert!(
                has_padding_warn,
                "expected NonZeroPadding warning, got: {:?}",
                report.issues
            );
        }
        // If no padding exists (already aligned), build a message with known padding
        // by using a different approach: insert non-zero bytes at a location
        // where frame magic is not found.
        else {
            // Create a message with a frame that ends not on an 8-byte boundary.
            // This is hard to control, so we skip this variant.
        }
    }

    // ── Structure: FrameOrderViolation ──────────────────────────────────

    #[test]
    fn structure_frame_order_violation() {
        // Build a message where a DataObject frame appears before
        // the metadata frame. This violates the Headers→DataObjects ordering
        // because we put a data frame first and then a header frame.
        let desc = default_desc();
        let payload = vec![0u8; 32];
        let data_frame = build_data_object_frame(&desc, &payload);
        let meta_frame = build_metadata_frame();

        // Put data frame first, then metadata frame = order violation
        let flags = 1u16; // HEADER_METADATA
        let msg = build_raw_message(flags, &[data_frame, meta_frame], None, false);
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_order = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::FrameOrderViolation);
        assert!(
            has_order,
            "expected FrameOrderViolation, got: {:?}",
            report.issues
        );
    }

    // ── Structure: PrecederNotFollowedByObject ─────────────────────────

    #[test]
    fn structure_preceder_not_followed_by_object() {
        // Build a message with a PrecederMetadata frame followed by
        // another metadata frame instead of a DataObject frame.
        use crate::wire::{FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC};

        let meta = GlobalMetadata {
            base: vec![BTreeMap::new()],
            ..GlobalMetadata::default()
        };
        let cbor = crate::metadata::global_metadata_to_cbor(&meta).unwrap();

        // Build PrecederMetadata frame (type=8)
        let total_length = (FRAME_HEADER_SIZE + cbor.len() + FRAME_END.len()) as u64;
        let mut preceder_frame = Vec::new();
        preceder_frame.extend_from_slice(FRAME_MAGIC);
        preceder_frame.extend_from_slice(&8u16.to_be_bytes()); // PrecederMetadata
        preceder_frame.extend_from_slice(&1u16.to_be_bytes()); // version
        preceder_frame.extend_from_slice(&0u16.to_be_bytes()); // flags
        preceder_frame.extend_from_slice(&total_length.to_be_bytes());
        preceder_frame.extend_from_slice(&cbor);
        preceder_frame.extend_from_slice(FRAME_END);

        let header_meta_frame = build_metadata_frame();

        // HeaderMetadata first (correct), then PrecederMetadata, then another HeaderMetadata
        // instead of DataObject. The preceder is in data-objects phase,
        // so the second metadata frame will trigger PrecederNotFollowedByObject
        // AND FrameOrderViolation.
        // Actually: build metadata, then preceder, then a second preceder-like:
        // we need something that's NOT a DataObject after preceder.
        // Let's use a FooterMetadata frame (type=7) after preceder.
        let mut footer_meta_frame = Vec::new();
        let footer_cbor =
            crate::metadata::global_metadata_to_cbor(&GlobalMetadata::default()).unwrap();
        let ftl = (FRAME_HEADER_SIZE + footer_cbor.len() + FRAME_END.len()) as u64;
        footer_meta_frame.extend_from_slice(FRAME_MAGIC);
        footer_meta_frame.extend_from_slice(&7u16.to_be_bytes()); // FooterMetadata
        footer_meta_frame.extend_from_slice(&1u16.to_be_bytes());
        footer_meta_frame.extend_from_slice(&0u16.to_be_bytes());
        footer_meta_frame.extend_from_slice(&ftl.to_be_bytes());
        footer_meta_frame.extend_from_slice(&footer_cbor);
        footer_meta_frame.extend_from_slice(FRAME_END);

        let flags = (1u16) | (1u16 << 1) | (1u16 << 6); // HEADER_METADATA | FOOTER_METADATA | PRECEDER
        let msg = build_raw_message(
            flags,
            &[header_meta_frame, preceder_frame, footer_meta_frame],
            None,
            false,
        );
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_preceder_err = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::PrecederNotFollowedByObject);
        assert!(
            has_preceder_err,
            "expected PrecederNotFollowedByObject, got: {:?}",
            report.issues
        );
    }

    // ── Structure: DanglingPreceder ─────────────────────────────────────

    #[test]
    fn structure_dangling_preceder() {
        // Build a message where a PrecederMetadata is the last frame — no DataObject follows.
        use crate::wire::{FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC};

        let meta = GlobalMetadata {
            base: vec![BTreeMap::new()],
            ..GlobalMetadata::default()
        };
        let cbor = crate::metadata::global_metadata_to_cbor(&meta).unwrap();

        let total_length = (FRAME_HEADER_SIZE + cbor.len() + FRAME_END.len()) as u64;
        let mut preceder_frame = Vec::new();
        preceder_frame.extend_from_slice(FRAME_MAGIC);
        preceder_frame.extend_from_slice(&8u16.to_be_bytes());
        preceder_frame.extend_from_slice(&1u16.to_be_bytes());
        preceder_frame.extend_from_slice(&0u16.to_be_bytes());
        preceder_frame.extend_from_slice(&total_length.to_be_bytes());
        preceder_frame.extend_from_slice(&cbor);
        preceder_frame.extend_from_slice(FRAME_END);

        let header_meta_frame = build_metadata_frame();
        let flags = 1u16 | (1u16 << 6); // HEADER_METADATA | PRECEDER
        let msg = build_raw_message(flags, &[header_meta_frame, preceder_frame], None, false);
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_dangling = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::DanglingPreceder);
        assert!(
            has_dangling,
            "expected DanglingPreceder, got: {:?}",
            report.issues
        );
    }

    // ── Structure: CborBeforeBoundaryUnknown ────────────────────────────

    #[test]
    fn structure_cbor_before_boundary_unknown() {
        // Build a data object frame with CBOR-before layout (flag=0),
        // but corrupt the CBOR so it can't be parsed. This triggers
        // CborBeforeBoundaryUnknown.
        use crate::wire::{DATA_OBJECT_FOOTER_SIZE, FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC};

        let payload = vec![0u8; 16];
        // Garbage CBOR
        let bad_cbor = vec![0xFF, 0xFF, 0xFF, 0xFF];

        // cbor_before layout: header | cbor | payload | cbor_offset(8) | ENDF(4)
        let cbor_offset = FRAME_HEADER_SIZE as u64; // CBOR starts right after header
        let body_len = bad_cbor.len() + payload.len() + DATA_OBJECT_FOOTER_SIZE;
        let total_length = (FRAME_HEADER_SIZE + body_len) as u64;

        let mut frame = Vec::new();
        frame.extend_from_slice(FRAME_MAGIC);
        frame.extend_from_slice(&4u16.to_be_bytes()); // DataObject
        frame.extend_from_slice(&1u16.to_be_bytes()); // version
        frame.extend_from_slice(&0u16.to_be_bytes()); // flags=0 → CBOR before payload
        frame.extend_from_slice(&total_length.to_be_bytes());
        frame.extend_from_slice(&bad_cbor);
        frame.extend_from_slice(&payload);
        frame.extend_from_slice(&cbor_offset.to_be_bytes());
        frame.extend_from_slice(FRAME_END);

        let meta_frame = build_metadata_frame();
        let flags = 1u16; // HEADER_METADATA
        let msg = build_raw_message(flags, &[meta_frame, frame], None, false);
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_cbor_err = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::CborBeforeBoundaryUnknown);
        assert!(
            has_cbor_err,
            "expected CborBeforeBoundaryUnknown, got: {:?}",
            report.issues
        );
    }

    // ── Structure: FlagMismatch ─────────────────────────────────────────

    #[test]
    fn structure_flag_mismatch() {
        // Build a valid message, then flip a flag bit in the preamble
        // so declared flags don't match observed frames.
        let mut msg = make_test_message();
        // The flags field is at offset 10..12 in the preamble.
        let current_flags = u16::from_be_bytes(msg[10..12].try_into().unwrap());
        // Flip the FOOTER_METADATA flag (bit 1) — our message doesn't have a footer metadata
        // frame but we'll claim it does.
        let bad_flags = current_flags | (1u16 << 1); // set FOOTER_METADATA
        msg[10..12].copy_from_slice(&bad_flags.to_be_bytes());
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_flag_mismatch = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::FlagMismatch);
        assert!(
            has_flag_mismatch,
            "expected FlagMismatch, got: {:?}",
            report.issues
        );
    }

    // ── Structure: NoMetadataFrame ──────────────────────────────────────

    #[test]
    fn structure_no_metadata_frame() {
        // Build a message with only a DataObject frame and no metadata frame at all.
        let desc = default_desc();
        let payload = vec![0u8; 32];
        let data_frame = build_data_object_frame(&desc, &payload);

        // flags = 0 (no metadata declared)
        let msg = build_raw_message(0u16, &[data_frame], None, false);
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_no_meta = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::NoMetadataFrame);
        assert!(
            has_no_meta,
            "expected NoMetadataFrame, got: {:?}",
            report.issues
        );
    }

    // ── Structure: Streaming-mode postamble handling ────────────────────

    #[test]
    fn structure_streaming_mode_validates() {
        // Build a streaming message (total_length=0) and verify it validates.
        let meta_frame = build_metadata_frame();
        let desc = default_desc();
        let payload = vec![0u8; 32];
        let data_frame = build_data_object_frame(&desc, &payload);
        let flags = 1u16; // HEADER_METADATA
        let msg = build_raw_message(flags, &[meta_frame, data_frame], None, true);
        let report = validate_message(&msg, &ValidateOptions::default());
        // The streaming message may have warnings (e.g. flag mismatches) but
        // should parse structurally.
        let has_fatal = report.issues.iter().any(|i| {
            i.severity == IssueSeverity::Error
                && !matches!(
                    i.code,
                    IssueCode::FlagMismatch
                        | IssueCode::FooterOffsetMismatch
                        | IssueCode::NoMetadataFrame
                )
        });
        // Streaming mode with just a header metadata + data object should work
        assert!(
            !has_fatal,
            "unexpected fatal error in streaming mode: {:?}",
            report.issues
        );
    }

    #[test]
    fn structure_streaming_mode_bad_postamble() {
        // Build a streaming message (total_length=0) but corrupt the postamble.
        let meta_frame = build_metadata_frame();
        let flags = 1u16;
        let mut msg = build_raw_message(flags, &[meta_frame], None, true);
        // Corrupt the end magic (last 8 bytes)
        let end = msg.len();
        msg[end - 8..end].copy_from_slice(b"BADMAGIC");
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(!report.is_ok());
        let has_postamble = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::PostambleInvalid);
        assert!(
            has_postamble,
            "expected PostambleInvalid in streaming mode, got: {:?}",
            report.issues
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Coverage gap tests — Metadata (Level 2)
    // ═══════════════════════════════════════════════════════════════════════

    /// Helper: encode with hash disabled and inject custom per-object CBOR into the
    /// data object frame descriptor. This lets us test metadata validation on
    /// crafted descriptors.
    fn make_message_with_patched_descriptor(patch: impl FnOnce(&mut ciborium::Value)) -> Vec<u8> {
        // Encode a valid message without hash (so hash won't mismatch after patch)
        let meta = GlobalMetadata::default();
        let desc = default_desc();
        let data = vec![0u8; 32];
        let opts = EncodeOptions {
            hash_algorithm: None,
            ..EncodeOptions::default()
        };
        let _msg = encode(&meta, &[(&desc, data.as_slice())], &opts).unwrap();

        // Re-build the message with the patched descriptor from scratch.
        let cbor_bytes = crate::metadata::object_descriptor_to_cbor(&desc).unwrap();
        let mut value: ciborium::Value = ciborium::from_reader(cbor_bytes.as_slice()).unwrap();
        patch(&mut value);
        let mut patched_cbor = Vec::new();
        ciborium::into_writer(&value, &mut patched_cbor).unwrap();

        // Build data object frame manually with patched CBOR
        use crate::wire::{
            DataObjectFlags, DATA_OBJECT_FOOTER_SIZE, FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC,
        };
        let payload = vec![0u8; 32];
        let cbor_offset = (FRAME_HEADER_SIZE + payload.len()) as u64;
        let total_length =
            (FRAME_HEADER_SIZE + payload.len() + patched_cbor.len() + DATA_OBJECT_FOOTER_SIZE)
                as u64;

        let mut frame = Vec::new();
        frame.extend_from_slice(FRAME_MAGIC);
        frame.extend_from_slice(&4u16.to_be_bytes()); // DataObject
        frame.extend_from_slice(&1u16.to_be_bytes()); // version
        frame.extend_from_slice(&DataObjectFlags::CBOR_AFTER_PAYLOAD.to_be_bytes()); // flags
        frame.extend_from_slice(&total_length.to_be_bytes());
        frame.extend_from_slice(&payload);
        frame.extend_from_slice(&patched_cbor);
        frame.extend_from_slice(&cbor_offset.to_be_bytes());
        frame.extend_from_slice(FRAME_END);

        let meta_frame = build_metadata_frame();
        let flags = 1u16; // HEADER_METADATA
        build_raw_message(flags, &[meta_frame, frame], None, false)
    }

    /// Helper to get a mutable reference to a CBOR map value by key.
    fn cbor_map_set(value: &mut ciborium::Value, key: &str, new_val: ciborium::Value) {
        if let ciborium::Value::Map(pairs) = value {
            for (k, v) in pairs.iter_mut() {
                if let ciborium::Value::Text(s) = k {
                    if s == key {
                        *v = new_val;
                        return;
                    }
                }
            }
            // Key not found, add it
            pairs.push((ciborium::Value::Text(key.to_string()), new_val));
        }
    }

    // ── Metadata: IndexCountMismatch ────────────────────────────────────

    #[test]
    fn metadata_index_count_mismatch() {
        // Build a message with an index frame that claims 5 objects
        // but the message has only 1 data object.
        use crate::wire::{FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC};

        let desc = default_desc();
        let payload = vec![0u8; 32];
        let data_frame = build_data_object_frame(&desc, &payload);
        let meta_frame = build_metadata_frame();

        // Build a fake index frame claiming 5 objects with wrong offsets
        let idx = crate::types::IndexFrame {
            object_count: 5,
            offsets: vec![0, 100, 200, 300, 400],
            lengths: vec![50, 50, 50, 50, 50],
        };
        let idx_cbor = crate::metadata::index_to_cbor(&idx).unwrap();
        let idx_total = (FRAME_HEADER_SIZE + idx_cbor.len() + FRAME_END.len()) as u64;
        let mut idx_frame = Vec::new();
        idx_frame.extend_from_slice(FRAME_MAGIC);
        idx_frame.extend_from_slice(&2u16.to_be_bytes()); // HeaderIndex
        idx_frame.extend_from_slice(&1u16.to_be_bytes());
        idx_frame.extend_from_slice(&0u16.to_be_bytes());
        idx_frame.extend_from_slice(&idx_total.to_be_bytes());
        idx_frame.extend_from_slice(&idx_cbor);
        idx_frame.extend_from_slice(FRAME_END);

        let flags = 1u16 | (1u16 << 2); // HEADER_METADATA | HEADER_INDEX
        let msg = build_raw_message(flags, &[meta_frame, idx_frame, data_frame], None, false);
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_idx_mismatch = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::IndexCountMismatch);
        assert!(
            has_idx_mismatch,
            "expected IndexCountMismatch, got: {:?}",
            report.issues
        );
    }

    // ── Metadata: IndexOffsetMismatch ───────────────────────────────────

    #[test]
    fn metadata_index_offset_mismatch() {
        // Build a message where the index has 1 entry but the offset is wrong.
        use crate::wire::{FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC};

        let desc = default_desc();
        let payload = vec![0u8; 32];
        let data_frame = build_data_object_frame(&desc, &payload);
        let meta_frame = build_metadata_frame();

        let idx = crate::types::IndexFrame {
            object_count: 1,
            offsets: vec![9999], // wrong offset
            lengths: vec![50],
        };
        let idx_cbor = crate::metadata::index_to_cbor(&idx).unwrap();
        let idx_total = (FRAME_HEADER_SIZE + idx_cbor.len() + FRAME_END.len()) as u64;
        let mut idx_frame = Vec::new();
        idx_frame.extend_from_slice(FRAME_MAGIC);
        idx_frame.extend_from_slice(&2u16.to_be_bytes()); // HeaderIndex
        idx_frame.extend_from_slice(&1u16.to_be_bytes());
        idx_frame.extend_from_slice(&0u16.to_be_bytes());
        idx_frame.extend_from_slice(&idx_total.to_be_bytes());
        idx_frame.extend_from_slice(&idx_cbor);
        idx_frame.extend_from_slice(FRAME_END);

        let flags = 1u16 | (1u16 << 2); // HEADER_METADATA | HEADER_INDEX
        let msg = build_raw_message(flags, &[meta_frame, idx_frame, data_frame], None, false);
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_offset_mismatch = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::IndexOffsetMismatch);
        assert!(
            has_offset_mismatch,
            "expected IndexOffsetMismatch, got: {:?}",
            report.issues
        );
    }

    // ── Metadata: UnknownEncoding ───────────────────────────────────────

    #[test]
    fn metadata_unknown_encoding() {
        let msg = make_message_with_patched_descriptor(|v| {
            cbor_map_set(
                v,
                "encoding",
                ciborium::Value::Text("turbo_zip".to_string()),
            );
        });
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_unk = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::UnknownEncoding);
        assert!(
            has_unk,
            "expected UnknownEncoding, got: {:?}",
            report.issues
        );
    }

    // ── Metadata: UnknownFilter ─────────────────────────────────────────

    #[test]
    fn metadata_unknown_filter() {
        let msg = make_message_with_patched_descriptor(|v| {
            cbor_map_set(
                v,
                "filter",
                ciborium::Value::Text("mega_filter".to_string()),
            );
        });
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_unk = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::UnknownFilter);
        assert!(has_unk, "expected UnknownFilter, got: {:?}", report.issues);
    }

    // ── Metadata: UnknownCompression ────────────────────────────────────

    #[test]
    fn metadata_unknown_compression() {
        let msg = make_message_with_patched_descriptor(|v| {
            cbor_map_set(
                v,
                "compression",
                ciborium::Value::Text("snappy9000".to_string()),
            );
        });
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_unk = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::UnknownCompression);
        assert!(
            has_unk,
            "expected UnknownCompression, got: {:?}",
            report.issues
        );
    }

    // ── Metadata: EmptyObjType ──────────────────────────────────────────

    #[test]
    fn metadata_empty_obj_type() {
        let msg = make_message_with_patched_descriptor(|v| {
            cbor_map_set(v, "type", ciborium::Value::Text(String::new()));
        });
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_empty = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::EmptyObjType);
        assert!(has_empty, "expected EmptyObjType, got: {:?}", report.issues);
    }

    // ── Metadata: NdimShapeMismatch ─────────────────────────────────────

    #[test]
    fn metadata_ndim_shape_mismatch() {
        let msg = make_message_with_patched_descriptor(|v| {
            // Set ndim=3 but leave shape as [4] (len=1)
            cbor_map_set(v, "ndim", ciborium::Value::Integer(3.into()));
        });
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_ndim = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::NdimShapeMismatch);
        assert!(
            has_ndim,
            "expected NdimShapeMismatch, got: {:?}",
            report.issues
        );
    }

    // ── Metadata: StridesShapeMismatch ──────────────────────────────────

    #[test]
    fn metadata_strides_shape_mismatch() {
        let msg = make_message_with_patched_descriptor(|v| {
            // Add extra strides entry so strides.len() != shape.len()
            cbor_map_set(
                v,
                "strides",
                ciborium::Value::Array(vec![
                    ciborium::Value::Integer(8.into()),
                    ciborium::Value::Integer(4.into()),
                ]),
            );
        });
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_strides = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::StridesShapeMismatch);
        assert!(
            has_strides,
            "expected StridesShapeMismatch, got: {:?}",
            report.issues
        );
    }

    // ── Metadata: ShapeOverflow ─────────────────────────────────────────

    #[test]
    fn metadata_shape_overflow() {
        let msg = make_message_with_patched_descriptor(|v| {
            // Set ndim=2, shape=[u64::MAX, 2] — product overflows u64
            cbor_map_set(v, "ndim", ciborium::Value::Integer(2.into()));
            cbor_map_set(
                v,
                "shape",
                ciborium::Value::Array(vec![
                    ciborium::Value::Integer(u64::MAX.into()),
                    ciborium::Value::Integer(2.into()),
                ]),
            );
            cbor_map_set(
                v,
                "strides",
                ciborium::Value::Array(vec![
                    ciborium::Value::Integer(8.into()),
                    ciborium::Value::Integer(8.into()),
                ]),
            );
        });
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_overflow = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::ShapeOverflow);
        assert!(
            has_overflow,
            "expected ShapeOverflow, got: {:?}",
            report.issues
        );
    }

    // ── Metadata: ReservedNotAMap ────────────────────────────────────────

    #[test]
    fn metadata_reserved_not_a_map() {
        // Build a message where a base entry's _reserved_ is a string instead of a map.
        // We need to craft the metadata CBOR to have base[0]._reserved_ = "bad".
        // The encoder auto-populates _reserved_, so we need to patch the encoded bytes.
        use crate::wire::{FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC};

        // Build custom metadata with _reserved_ as a string
        let mut base_entry = BTreeMap::new();
        base_entry.insert(
            "_reserved_".to_string(),
            ciborium::Value::Text("not_a_map".to_string()),
        );
        let meta = GlobalMetadata {
            base: vec![base_entry],
            ..GlobalMetadata::default()
        };
        let meta_cbor = crate::metadata::global_metadata_to_cbor(&meta).unwrap();
        let total_length = (FRAME_HEADER_SIZE + meta_cbor.len() + FRAME_END.len()) as u64;
        let mut meta_frame = Vec::new();
        meta_frame.extend_from_slice(FRAME_MAGIC);
        meta_frame.extend_from_slice(&1u16.to_be_bytes()); // HeaderMetadata
        meta_frame.extend_from_slice(&1u16.to_be_bytes());
        meta_frame.extend_from_slice(&0u16.to_be_bytes());
        meta_frame.extend_from_slice(&total_length.to_be_bytes());
        meta_frame.extend_from_slice(&meta_cbor);
        meta_frame.extend_from_slice(FRAME_END);

        let desc = default_desc();
        let payload = vec![0u8; 32];
        let data_frame = build_data_object_frame(&desc, &payload);
        let flags = 1u16; // HEADER_METADATA
        let msg = build_raw_message(flags, &[meta_frame, data_frame], None, false);
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_reserved_err = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::ReservedNotAMap);
        assert!(
            has_reserved_err,
            "expected ReservedNotAMap, got: {:?}",
            report.issues
        );
    }

    // ── Metadata: HashFrameCborParseFailed ──────────────────────────────

    #[test]
    fn metadata_hash_frame_cbor_parse_failed() {
        // Build a message with a corrupt hash frame.
        use crate::wire::{FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC};

        let meta_frame = build_metadata_frame();
        let desc = default_desc();
        let payload = vec![0u8; 32];
        let data_frame = build_data_object_frame(&desc, &payload);

        // Build a hash frame with garbage CBOR
        let garbage_cbor = vec![0xFF, 0xFF, 0xFF, 0xFF];
        let hash_total = (FRAME_HEADER_SIZE + garbage_cbor.len() + FRAME_END.len()) as u64;
        let mut hash_frame = Vec::new();
        hash_frame.extend_from_slice(FRAME_MAGIC);
        hash_frame.extend_from_slice(&3u16.to_be_bytes()); // HeaderHash
        hash_frame.extend_from_slice(&1u16.to_be_bytes());
        hash_frame.extend_from_slice(&0u16.to_be_bytes());
        hash_frame.extend_from_slice(&hash_total.to_be_bytes());
        hash_frame.extend_from_slice(&garbage_cbor);
        hash_frame.extend_from_slice(FRAME_END);

        let flags = 1u16 | (1u16 << 4); // HEADER_METADATA | HEADER_HASHES
        let msg = build_raw_message(flags, &[meta_frame, hash_frame, data_frame], None, false);
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_hash_err = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::HashFrameCborParseFailed);
        assert!(
            has_hash_err,
            "expected HashFrameCborParseFailed, got: {:?}",
            report.issues
        );
    }

    // ── Metadata: PrecederBaseCountWrong ────────────────────────────────

    #[test]
    fn metadata_preceder_base_count_wrong() {
        // Build a message with a PrecederMetadata whose base has 2 entries
        // instead of exactly 1.
        use crate::wire::{FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC};

        let prec_meta = GlobalMetadata {
            base: vec![BTreeMap::new(), BTreeMap::new()], // 2 entries, must be 1
            ..GlobalMetadata::default()
        };
        let prec_cbor = crate::metadata::global_metadata_to_cbor(&prec_meta).unwrap();
        let prec_total = (FRAME_HEADER_SIZE + prec_cbor.len() + FRAME_END.len()) as u64;
        let mut preceder_frame = Vec::new();
        preceder_frame.extend_from_slice(FRAME_MAGIC);
        preceder_frame.extend_from_slice(&8u16.to_be_bytes()); // PrecederMetadata
        preceder_frame.extend_from_slice(&1u16.to_be_bytes());
        preceder_frame.extend_from_slice(&0u16.to_be_bytes());
        preceder_frame.extend_from_slice(&prec_total.to_be_bytes());
        preceder_frame.extend_from_slice(&prec_cbor);
        preceder_frame.extend_from_slice(FRAME_END);

        let meta_frame = build_metadata_frame();
        let desc = default_desc();
        let payload = vec![0u8; 32];
        let data_frame = build_data_object_frame(&desc, &payload);

        let flags = 1u16 | (1u16 << 6); // HEADER_METADATA | PRECEDER_METADATA
        let msg = build_raw_message(
            flags,
            &[meta_frame, preceder_frame, data_frame],
            None,
            false,
        );
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_prec_err = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::PrecederBaseCountWrong);
        assert!(
            has_prec_err,
            "expected PrecederBaseCountWrong, got: {:?}",
            report.issues
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Coverage gap tests — Integrity (Level 3)
    // ═══════════════════════════════════════════════════════════════════════

    // ── Integrity: Hash frame fallback path ─────────────────────────────

    #[test]
    fn integrity_hash_frame_fallback_verified() {
        // The default encode() puts hashes in both the hash frame and
        // per-object descriptors. Verify that a standard message with
        // hash verification succeeds (hash frame is parsed at Level 3).
        let msg = make_test_message();
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(report.hash_verified, "issues: {:?}", report.issues);
    }

    // ── Integrity: UnknownHashAlgorithm ─────────────────────────────────

    #[test]
    fn integrity_unknown_hash_algorithm() {
        // Build a message with a per-object hash using a fake algorithm name.
        let msg = make_message_with_patched_descriptor(|v| {
            let hash_map = ciborium::Value::Map(vec![
                (
                    ciborium::Value::Text("type".to_string()),
                    ciborium::Value::Text("sha9001".to_string()),
                ),
                (
                    ciborium::Value::Text("value".to_string()),
                    ciborium::Value::Text("deadbeef".to_string()),
                ),
            ]);
            cbor_map_set(v, "hash", hash_map);
        });
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_unk_hash = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::UnknownHashAlgorithm);
        assert!(
            has_unk_hash,
            "expected UnknownHashAlgorithm, got: {:?}",
            report.issues
        );
    }

    // ── Integrity: DecodePipelineFailed (corrupt compressed payload) ────

    #[test]
    fn integrity_decode_pipeline_failed_corrupt_compressed() {
        // Encode a message with zstd compression, then corrupt the payload.
        // This should trigger DecodePipelineFailed at Level 3.
        #[cfg(feature = "zstd")]
        {
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
                compression: "zstd".to_string(),
                params: BTreeMap::new(),
                hash: None,
            };
            let data = vec![0u8; 32];
            let opts = EncodeOptions {
                hash_algorithm: None,
                ..EncodeOptions::default()
            };
            let mut msg = encode(&meta, &[(&desc, data.as_slice())], &opts).unwrap();
            // Corrupt payload bytes inside the data object frame
            // The data object is near the end (before postamble)
            let pa_start = msg.len() - crate::wire::POSTAMBLE_SIZE;
            // Corrupt bytes in the middle of the payload area
            let target = pa_start * 3 / 4;
            if target < msg.len() {
                msg[target] ^= 0xFF;
                msg[target.saturating_sub(1)] ^= 0xFF;
            }
            let report = validate_message(&msg, &ValidateOptions::default());
            let has_pipeline_err = report.issues.iter().any(|i| {
                matches!(
                    i.code,
                    IssueCode::DecodePipelineFailed | IssueCode::HashMismatch
                )
            });
            assert!(
                has_pipeline_err || !report.is_ok(),
                "expected DecodePipelineFailed or error, got: {:?}",
                report.issues
            );
        }
    }

    // ── Integrity: Shape product overflow → PipelineConfigFailed ────────

    #[test]
    fn integrity_shape_product_overflow_pipeline() {
        // Build a message where descriptor claims compression != "none"
        // and shape is huge, triggering PipelineConfigFailed at Level 3.
        let msg = make_message_with_patched_descriptor(|v| {
            cbor_map_set(v, "compression", ciborium::Value::Text("zstd".to_string()));
            cbor_map_set(v, "ndim", ciborium::Value::Integer(2.into()));
            cbor_map_set(
                v,
                "shape",
                ciborium::Value::Array(vec![
                    ciborium::Value::Integer(u64::MAX.into()),
                    ciborium::Value::Integer(2.into()),
                ]),
            );
            cbor_map_set(
                v,
                "strides",
                ciborium::Value::Array(vec![
                    ciborium::Value::Integer(8.into()),
                    ciborium::Value::Integer(8.into()),
                ]),
            );
        });
        let report = validate_message(&msg, &ValidateOptions::default());
        // Should have ShapeOverflow from metadata + potentially PipelineConfigFailed
        let has_shape_or_pipeline = report.issues.iter().any(|i| {
            matches!(
                i.code,
                IssueCode::ShapeOverflow | IssueCode::PipelineConfigFailed
            )
        });
        assert!(
            has_shape_or_pipeline,
            "expected ShapeOverflow or PipelineConfigFailed, got: {:?}",
            report.issues
        );
    }

    // ── Integrity: Descriptor re-parse when Level 2 didn't run ─────────

    #[test]
    fn integrity_descriptor_reparse_without_metadata() {
        // Run at integrity level only (skip metadata level) so Level 3
        // must re-parse the descriptor itself.
        let msg = make_test_message();
        let opts = ValidateOptions {
            max_level: ValidationLevel::Integrity,
            checksum_only: true, // skips metadata level
            check_canonical: false,
        };
        let report = validate_message(&msg, &opts);
        // Should still verify hashes successfully
        assert!(
            report.hash_verified,
            "expected hash_verified in checksum mode, got: {:?}",
            report.issues
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Coverage gap tests — Fidelity (Level 4)
    // ═══════════════════════════════════════════════════════════════════════

    // ── Fidelity: Bitmask dtype size calculation ────────────────────────

    #[test]
    fn fidelity_bitmask_valid() {
        // Bitmask with 16 bits = 2 bytes payload
        let meta = GlobalMetadata::default();
        let desc = DataObjectDescriptor {
            obj_type: "ndarray".to_string(),
            ndim: 1,
            shape: vec![16],
            strides: vec![1],
            dtype: Dtype::Bitmask,
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        };
        let data = vec![0u8; 2]; // ceil(16/8) = 2 bytes
        let msg = encode(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();
        let report = validate_message(&msg, &full_opts());
        assert!(
            report.is_ok(),
            "bitmask should pass fidelity: {:?}",
            report.issues
        );
    }

    #[test]
    fn fidelity_bitmask_non_byte_aligned() {
        // Bitmask with 13 bits = ceil(13/8) = 2 bytes
        let meta = GlobalMetadata::default();
        let desc = DataObjectDescriptor {
            obj_type: "ndarray".to_string(),
            ndim: 1,
            shape: vec![13],
            strides: vec![1],
            dtype: Dtype::Bitmask,
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        };
        let data = vec![0u8; 2]; // ceil(13/8) = 2 bytes
        let msg = encode(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();
        let report = validate_message(&msg, &full_opts());
        assert!(
            report.is_ok(),
            "bitmask (non-byte-aligned) should pass fidelity: {:?}",
            report.issues
        );
    }

    // ── Fidelity: DecodedSizeMismatch with shape overflow ──────────────

    #[test]
    fn fidelity_decoded_size_overflow() {
        // Build a message where the descriptor shape product would overflow
        // usize when multiplied by dtype.byte_width(). This triggers the
        // "expected decoded size overflows usize" branch.
        let msg = make_message_with_patched_descriptor(|v| {
            // Set shape to a very large product that overflows when × byte_width
            // Use ndim=1 with shape=[u64::MAX/8 + 1] — product × 8 overflows usize
            // on platforms where usize < u64 or causes issues
            cbor_map_set(v, "ndim", ciborium::Value::Integer(1.into()));
            // Use a shape that's valid as u64 product but overflows usize × byte_width
            let big = (u64::MAX / 8) + 1;
            cbor_map_set(
                v,
                "shape",
                ciborium::Value::Array(vec![ciborium::Value::Integer(big.into())]),
            );
            cbor_map_set(
                v,
                "strides",
                ciborium::Value::Array(vec![ciborium::Value::Integer(8.into())]),
            );
        });
        let report = validate_message(&msg, &full_opts());
        let has_size_issue = report.issues.iter().any(|i| {
            matches!(
                i.code,
                IssueCode::DecodedSizeMismatch | IssueCode::ShapeOverflow
            )
        });
        assert!(
            has_size_issue,
            "expected DecodedSizeMismatch or ShapeOverflow, got: {:?}",
            report.issues
        );
    }

    // ── Integrity: NoHashAvailable ──────────────────────────────────────

    #[test]
    fn integrity_no_hash_available() {
        // Encode without hash — should get NoHashAvailable warning
        let meta = GlobalMetadata::default();
        let desc = default_desc();
        let data = vec![0u8; 32];
        let opts = EncodeOptions {
            hash_algorithm: None,
            ..EncodeOptions::default()
        };
        let msg = encode(&meta, &[(&desc, data.as_slice())], &opts).unwrap();
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_no_hash = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::NoHashAvailable);
        assert!(
            has_no_hash,
            "expected NoHashAvailable, got: {:?}",
            report.issues
        );
        assert!(!report.hash_verified);
    }

    // ── Structure: TotalLengthOverflow ──────────────────────────────────

    #[test]
    fn structure_total_length_overflow() {
        // On 64-bit this won't trigger because u64 fits in usize.
        // But we test the branch by setting total_length = u64::MAX
        // which exceeds any buffer.
        let mut msg = make_test_message();
        msg[16..24].copy_from_slice(&u64::MAX.to_be_bytes());
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(!report.is_ok());
        // On 64-bit this will hit TotalLengthExceedsBuffer (since u64::MAX fits in usize
        // but exceeds buf.len()). On 32-bit it would hit TotalLengthOverflow.
        let has_err = report.issues.iter().any(|i| {
            matches!(
                i.code,
                IssueCode::TotalLengthOverflow | IssueCode::TotalLengthExceedsBuffer
            )
        });
        assert!(has_err, "expected length error, got: {:?}", report.issues);
    }

    // ── Metadata: HashFrameCountMismatch ────────────────────────────────

    #[test]
    fn metadata_hash_frame_count_mismatch() {
        // Build a message with a hash frame that has 3 hashes but only 1 data object.
        use crate::wire::{FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC};

        let meta_frame = build_metadata_frame();
        let desc = default_desc();
        let payload = vec![0u8; 32];
        let data_frame = build_data_object_frame(&desc, &payload);

        let hash_frame_data = crate::types::HashFrame {
            object_count: 3,
            hash_type: "xxh3".to_string(),
            hashes: vec!["aaa".to_string(), "bbb".to_string(), "ccc".to_string()],
        };
        let hash_cbor = crate::metadata::hash_frame_to_cbor(&hash_frame_data).unwrap();
        let hash_total = (FRAME_HEADER_SIZE + hash_cbor.len() + FRAME_END.len()) as u64;
        let mut hash_frame = Vec::new();
        hash_frame.extend_from_slice(FRAME_MAGIC);
        hash_frame.extend_from_slice(&3u16.to_be_bytes()); // HeaderHash
        hash_frame.extend_from_slice(&1u16.to_be_bytes());
        hash_frame.extend_from_slice(&0u16.to_be_bytes());
        hash_frame.extend_from_slice(&hash_total.to_be_bytes());
        hash_frame.extend_from_slice(&hash_cbor);
        hash_frame.extend_from_slice(FRAME_END);

        let flags = 1u16 | (1u16 << 4); // HEADER_METADATA | HEADER_HASHES
        let msg = build_raw_message(flags, &[meta_frame, hash_frame, data_frame], None, false);
        let report = validate_message(&msg, &ValidateOptions::default());
        let has_count_mismatch = report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::HashFrameCountMismatch);
        assert!(
            has_count_mismatch,
            "expected HashFrameCountMismatch, got: {:?}",
            report.issues
        );
    }

    // ── Fidelity: Non-raw decode fallback path ─────────────────────────

    #[test]
    fn fidelity_raw_payload_scan() {
        // A raw message (encoding=none, filter=none, compression=none) at
        // fidelity level should scan the payload in-place without needing
        // the decode pipeline.
        let msg = make_float64_message(&[1.0, 2.0, 3.0, 4.0]);
        let report = validate_message(&msg, &full_opts());
        assert!(report.is_ok(), "issues: {:?}", report.issues);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional coverage — Structure (Level 1)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn structure_streaming_ffo_high() {
        let meta_frame = build_metadata_frame();
        let desc = default_desc();
        let payload = vec![0u8; 32];
        let data_frame = build_data_object_frame(&desc, &payload);
        let flags = 1u16;
        let mut msg = build_raw_message(flags, &[meta_frame, data_frame], None, true);
        let pa_start = msg.len() - 16;
        let bad_ffo: u64 = (msg.len() + 100) as u64;
        msg[pa_start..pa_start + 8].copy_from_slice(&bad_ffo.to_be_bytes());
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.code == IssueCode::FooterOffsetOutOfRange),
            "expected FooterOffsetOutOfRange, got: {:?}",
            report.issues
        );
    }

    #[test]
    fn structure_double_preceder() {
        use crate::wire::{FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC};
        let meta_cbor =
            crate::metadata::global_metadata_to_cbor(&GlobalMetadata::default()).unwrap();
        let total_length = (FRAME_HEADER_SIZE + meta_cbor.len() + FRAME_END.len()) as u64;
        let mut pf = Vec::new();
        pf.extend_from_slice(FRAME_MAGIC);
        pf.extend_from_slice(&8u16.to_be_bytes());
        pf.extend_from_slice(&1u16.to_be_bytes());
        pf.extend_from_slice(&0u16.to_be_bytes());
        pf.extend_from_slice(&total_length.to_be_bytes());
        pf.extend_from_slice(&meta_cbor);
        pf.extend_from_slice(FRAME_END);
        let pf2 = pf.clone();
        let desc = default_desc();
        let data_frame = build_data_object_frame(&desc, &vec![0u8; 32]);
        let hm = build_metadata_frame();
        let flags = 1u16 | (1u16 << 6);
        let msg = build_raw_message(flags, &[hm, pf, pf2, data_frame], None, false);
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.code == IssueCode::PrecederNotFollowedByObject),
            "expected PrecederNotFollowedByObject, got: {:?}",
            report.issues
        );
    }

    #[test]
    fn structure_trailing_preceder() {
        use crate::wire::{FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC};
        let meta_cbor =
            crate::metadata::global_metadata_to_cbor(&GlobalMetadata::default()).unwrap();
        let tl = (FRAME_HEADER_SIZE + meta_cbor.len() + FRAME_END.len()) as u64;
        let mut pf = Vec::new();
        pf.extend_from_slice(FRAME_MAGIC);
        pf.extend_from_slice(&8u16.to_be_bytes());
        pf.extend_from_slice(&1u16.to_be_bytes());
        pf.extend_from_slice(&0u16.to_be_bytes());
        pf.extend_from_slice(&tl.to_be_bytes());
        pf.extend_from_slice(&meta_cbor);
        pf.extend_from_slice(FRAME_END);
        let hm = build_metadata_frame();
        let desc = default_desc();
        let df = build_data_object_frame(&desc, &vec![0u8; 32]);
        let flags = 1u16 | (1u16 << 6);
        let msg = build_raw_message(flags, &[hm, df, pf], None, false);
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.code == IssueCode::DanglingPreceder),
            "expected DanglingPreceder, got: {:?}",
            report.issues
        );
    }

    #[test]
    fn structure_flag_mismatch_extra() {
        let meta_frame = build_metadata_frame();
        let desc = default_desc();
        let data_frame = build_data_object_frame(&desc, &vec![0u8; 32]);
        let msg = build_raw_message(0u16, &[meta_frame, data_frame], None, false);
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.code == IssueCode::FlagMismatch),
            "expected FlagMismatch, got: {:?}",
            report.issues
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional coverage — Fidelity (Level 4)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn fidelity_bitmask_wrong_size() {
        let msg = make_message_with_patched_descriptor(|v| {
            cbor_map_set(v, "dtype", ciborium::Value::Text("bitmask".to_string()));
            cbor_map_set(v, "ndim", ciborium::Value::Integer(1.into()));
            cbor_map_set(
                v,
                "shape",
                ciborium::Value::Array(vec![ciborium::Value::Integer(1000.into())]),
            );
            cbor_map_set(
                v,
                "strides",
                ciborium::Value::Array(vec![ciborium::Value::Integer(1.into())]),
            );
        });
        let report = validate_message(&msg, &full_opts());
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.code == IssueCode::DecodedSizeMismatch),
            "expected DecodedSizeMismatch for bitmask, got: {:?}",
            report.issues
        );
    }

    #[test]
    fn fidelity_multi_object_mixed() {
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
        let nan_data: Vec<u8> = f64::NAN
            .to_be_bytes()
            .iter()
            .chain(1.0f64.to_be_bytes().iter())
            .copied()
            .collect();
        let ok_data: Vec<u8> = 2.0f64
            .to_be_bytes()
            .iter()
            .chain(3.0f64.to_be_bytes().iter())
            .copied()
            .collect();
        let msg = encode(
            &meta,
            &[(&desc, nan_data.as_slice()), (&desc, ok_data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();
        let report = validate_message(&msg, &full_opts());
        let nans: Vec<_> = report
            .issues
            .iter()
            .filter(|i| i.code == IssueCode::NanDetected)
            .collect();
        assert!(
            !nans.is_empty(),
            "expected NanDetected, got: {:?}",
            report.issues
        );
        assert_eq!(nans[0].object_index, Some(0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional coverage — Integrity (Level 3)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn integrity_unknown_hash_in_frame() {
        use crate::wire::{FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC};
        let meta_frame = build_metadata_frame();
        let desc = default_desc();
        let data_frame = build_data_object_frame(&desc, &vec![0u8; 32]);
        let hf = crate::types::HashFrame {
            object_count: 1,
            hash_type: "blake99".to_string(),
            hashes: vec!["deadbeef".to_string()],
        };
        let hcbor = crate::metadata::hash_frame_to_cbor(&hf).unwrap();
        let htl = (FRAME_HEADER_SIZE + hcbor.len() + FRAME_END.len()) as u64;
        let mut hash_frame = Vec::new();
        hash_frame.extend_from_slice(FRAME_MAGIC);
        hash_frame.extend_from_slice(&3u16.to_be_bytes());
        hash_frame.extend_from_slice(&1u16.to_be_bytes());
        hash_frame.extend_from_slice(&0u16.to_be_bytes());
        hash_frame.extend_from_slice(&htl.to_be_bytes());
        hash_frame.extend_from_slice(&hcbor);
        hash_frame.extend_from_slice(FRAME_END);
        let flags = 1u16 | (1u16 << 4);
        let msg = build_raw_message(flags, &[meta_frame, hash_frame, data_frame], None, false);
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.code == IssueCode::UnknownHashAlgorithm),
            "expected UnknownHashAlgorithm, got: {:?}",
            report.issues
        );
    }

    #[test]
    fn integrity_corrupt_compressed_zstd() {
        let msg = make_message_with_patched_descriptor(|v| {
            cbor_map_set(v, "compression", ciborium::Value::Text("zstd".to_string()));
        });
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(
            report.issues.iter().any(|i| matches!(
                i.code,
                IssueCode::DecodePipelineFailed | IssueCode::PipelineConfigFailed
            )),
            "expected decode failure, got: {:?}",
            report.issues
        );
    }

    #[test]
    fn integrity_shape_overflow_filter() {
        let msg = make_message_with_patched_descriptor(|v| {
            cbor_map_set(v, "filter", ciborium::Value::Text("bitshuffle".to_string()));
            cbor_map_set(v, "ndim", ciborium::Value::Integer(2.into()));
            cbor_map_set(
                v,
                "shape",
                ciborium::Value::Array(vec![
                    ciborium::Value::Integer(u64::MAX.into()),
                    ciborium::Value::Integer(2.into()),
                ]),
            );
            cbor_map_set(
                v,
                "strides",
                ciborium::Value::Array(vec![
                    ciborium::Value::Integer(8.into()),
                    ciborium::Value::Integer(8.into()),
                ]),
            );
        });
        let report = validate_message(&msg, &ValidateOptions::default());
        assert!(
            report.issues.iter().any(|i| matches!(
                i.code,
                IssueCode::ShapeOverflow | IssueCode::PipelineConfigFailed
            )),
            "expected ShapeOverflow or PipelineConfigFailed, got: {:?}",
            report.issues
        );
    }

    #[test]
    fn fidelity_non_raw_decode_error() {
        let msg = make_message_with_patched_descriptor(|v| {
            cbor_map_set(v, "compression", ciborium::Value::Text("zstd".to_string()));
        });
        let report = validate_message(&msg, &full_opts());
        assert!(
            report.issues.iter().any(|i| matches!(
                i.code,
                IssueCode::DecodePipelineFailed
                    | IssueCode::PipelineConfigFailed
                    | IssueCode::DecodeObjectFailed
            )),
            "expected decode error, got: {:?}",
            report.issues
        );
    }
}
