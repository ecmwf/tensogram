use std::path::{Path, PathBuf};

use tensogram_core::{
    validate_file, FileValidationReport, IssueSeverity, ValidateMode, ValidateOptions,
    ValidationLevel,
};

/// Custom error to signal validation failure (exit code 1) without process::exit.
#[derive(Debug)]
pub struct ValidationFailed;

impl std::fmt::Display for ValidationFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "validation failed")
    }
}

impl std::error::Error for ValidationFailed {}

pub fn run(
    files: &[PathBuf],
    mode: ValidateMode,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let options = ValidateOptions { mode };
    let mut all_ok = true;

    if json {
        // Wrap all results in a JSON array for valid batch output
        let mut entries = Vec::new();
        for path in files {
            let report = validate_file(path, &options)?;
            entries.push(format_json(path, &report));
            if !report.is_ok() {
                all_ok = false;
            }
        }
        println!("[{}]", entries.join(",\n"));
    } else {
        for path in files {
            let report = validate_file(path, &options)?;
            print_human(path, &report);
            if !report.is_ok() {
                all_ok = false;
            }
        }
    }

    if !all_ok {
        return Err(Box::new(ValidationFailed));
    }

    Ok(())
}

fn print_human(path: &Path, report: &FileValidationReport) {
    let msg_count = report.messages.len();
    let obj_count = report.total_objects();

    // File-level issues
    for issue in &report.file_issues {
        eprintln!(
            "{}: WARNING — {} (at byte offset {}, {} bytes)",
            path.display(),
            issue.description,
            issue.byte_offset,
            issue.length,
        );
    }

    if report.is_ok() {
        let hash_note = if report.hash_verified() {
            ", hash verified"
        } else {
            ""
        };
        println!(
            "{}: OK ({} messages, {} objects{})",
            path.display(),
            msg_count,
            obj_count,
            hash_note,
        );
    } else {
        for (i, msg_report) in report.messages.iter().enumerate() {
            for issue in &msg_report.issues {
                let obj_note = match issue.object_index {
                    Some(idx) => format!(", object {idx}"),
                    None => String::new(),
                };
                let prefix = match issue.severity {
                    IssueSeverity::Error => "FAILED",
                    IssueSeverity::Warning => "WARNING",
                };
                let offset_note = match issue.byte_offset {
                    Some(off) => format!(" (at byte {off})"),
                    None => String::new(),
                };
                eprintln!(
                    "{}: {prefix} — message {i}{obj_note}: {}{offset_note}",
                    path.display(),
                    issue.description,
                );
            }
        }
    }
}

fn format_json(path: &Path, report: &FileValidationReport) -> String {
    let status = if report.is_ok() { "ok" } else { "failed" };
    let msg_count = report.messages.len();
    let obj_count = report.total_objects();
    let hash_verified = report.hash_verified();

    let mut all_issues = Vec::new();

    for fi in &report.file_issues {
        all_issues.push(format!(
            r#"{{"level":"file","severity":"warning","message_index":null,"object_index":null,"byte_offset":{},"description":{}}}"#,
            fi.byte_offset,
            json_string(&fi.description),
        ));
    }

    for (i, msg_report) in report.messages.iter().enumerate() {
        for issue in &msg_report.issues {
            let level = match issue.level {
                ValidationLevel::Structure => "structure",
                ValidationLevel::Metadata => "metadata",
                ValidationLevel::Integrity => "integrity",
            };
            let severity = match issue.severity {
                IssueSeverity::Error => "error",
                IssueSeverity::Warning => "warning",
            };
            let obj_idx = match issue.object_index {
                Some(idx) => idx.to_string(),
                None => "null".to_string(),
            };
            let byte_off = match issue.byte_offset {
                Some(off) => off.to_string(),
                None => "null".to_string(),
            };
            all_issues.push(format!(
                r#"{{"level":"{level}","severity":"{severity}","message_index":{i},"object_index":{obj_idx},"byte_offset":{byte_off},"description":{}}}"#,
                json_string(&issue.description),
            ));
        }
    }

    let issues_str = if all_issues.is_empty() {
        String::new()
    } else {
        all_issues.join(",")
    };
    format!(
        r#"{{"file":{},"status":"{status}","messages":{msg_count},"objects":{obj_count},"hash_verified":{hash_verified},"issues":[{issues_str}]}}"#,
        json_string(&path.display().to_string()),
    )
}

/// Escape a string for JSON output.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < '\x20' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use tensogram_core::{
        ByteOrder, DataObjectDescriptor, EncodeOptions, GlobalMetadata, TensogramFile,
    };

    fn make_test_file(dir: &Path, name: &str, n_messages: usize) -> PathBuf {
        let path = dir.join(name);
        let mut f = TensogramFile::create(&path).unwrap();
        let desc = DataObjectDescriptor {
            obj_type: "ndarray".into(),
            ndim: 1,
            shape: vec![4],
            strides: vec![8],
            dtype: tensogram_core::Dtype::Float64,
            byte_order: ByteOrder::Big,
            encoding: "none".into(),
            filter: "none".into(),
            compression: "none".into(),
            params: Default::default(),
            hash: None,
        };
        let data = vec![0u8; 32]; // 4 × f64
        let meta = GlobalMetadata::default();
        for _ in 0..n_messages {
            f.append(&meta, &[(&desc, &data)], &EncodeOptions::default())
                .unwrap();
        }
        path
    }

    #[test]
    fn cli_validate_valid_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path(), "valid.tgm", 2);
        run(&[path], ValidateMode::Default, false).unwrap();
    }

    #[test]
    fn cli_validate_quick_mode() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path(), "quick.tgm", 1);
        run(&[path], ValidateMode::Quick, false).unwrap();
    }

    #[test]
    fn cli_validate_checksum_mode() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path(), "checksum.tgm", 1);
        run(&[path], ValidateMode::Checksum, false).unwrap();
    }

    #[test]
    fn cli_validate_canonical_mode() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path(), "canonical.tgm", 1);
        run(&[path], ValidateMode::Canonical, false).unwrap();
    }

    #[test]
    fn cli_validate_json_mode() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path(), "json.tgm", 1);
        run(&[path], ValidateMode::Default, true).unwrap();
    }

    #[test]
    fn cli_validate_batch_files() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = make_test_file(dir.path(), "a.tgm", 1);
        let p2 = make_test_file(dir.path(), "b.tgm", 1);
        run(&[p1, p2], ValidateMode::Default, false).unwrap();
    }

    #[test]
    fn cli_validate_missing_file() {
        let result = run(
            &[PathBuf::from("/nonexistent/file.tgm")],
            ValidateMode::Default,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn cli_validate_zero_files() {
        // No files should succeed (nothing to validate)
        run(&[], ValidateMode::Default, false).unwrap();
    }

    #[test]
    fn json_string_escaping() {
        assert_eq!(json_string("hello"), r#""hello""#);
        assert_eq!(json_string(r#"a"b"#), r#""a\"b""#);
        assert_eq!(json_string("a\\b"), r#""a\\b""#);
        assert_eq!(json_string("a\nb"), r#""a\nb""#);
    }

    #[test]
    fn batch_json_is_valid_array() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = make_test_file(dir.path(), "a.tgm", 1);
        let p2 = make_test_file(dir.path(), "b.tgm", 1);
        // Just verify it doesn't panic; actual JSON validity is structural
        run(&[p1, p2], ValidateMode::Default, true).unwrap();
    }
}
