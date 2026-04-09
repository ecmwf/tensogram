use std::path::{Path, PathBuf};

use serde::Serialize;
use tensogram_core::{
    validate_file, FileIssue, FileValidationReport, IssueSeverity, ValidateMode, ValidateOptions,
    ValidationReport,
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
        let mut entries: Vec<JsonFileReport> = Vec::new();
        for path in files {
            let report = validate_file(path, &options)?;
            if !report.is_ok() {
                all_ok = false;
            }
            entries.push(JsonFileReport::from_report(path, &report));
        }
        println!("{}", serde_json::to_string_pretty(&entries)?);
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

    // Print all issues: errors + warnings in failure mode, warnings only in OK mode
    let is_ok = report.is_ok();
    for (i, msg_report) in report.messages.iter().enumerate() {
        for issue in &msg_report.issues {
            if is_ok && issue.severity != IssueSeverity::Warning {
                continue;
            }
            let obj_note = match issue.object_index {
                Some(idx) => format!(", object {}", idx + 1),
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
                "{}: {prefix} — message {}{obj_note}: {}{offset_note}",
                path.display(),
                i + 1,
                issue.description,
            );
        }
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
        // Final summary line
        let msg_errors: usize = report
            .messages
            .iter()
            .flat_map(|r| &r.issues)
            .filter(|i| i.severity == IssueSeverity::Error)
            .count();
        let file_issue_count = report.file_issues.len();
        let total_problems = msg_errors + file_issue_count;
        eprintln!(
            "{}: FAILED ({} errors, {} messages, {} objects)",
            path.display(),
            total_problems,
            msg_count,
            obj_count,
        );
    }
}

// ── JSON output types (serde-driven) ────────────────────────────────────────

#[derive(Serialize)]
struct JsonFileReport {
    file: String,
    status: &'static str,
    messages: usize,
    objects: usize,
    hash_verified: bool,
    file_issues: Vec<FileIssue>,
    message_reports: Vec<ValidationReport>,
}

impl JsonFileReport {
    fn from_report(path: &Path, report: &FileValidationReport) -> Self {
        Self {
            file: path.display().to_string(),
            status: if report.is_ok() { "ok" } else { "failed" },
            messages: report.messages.len(),
            objects: report.total_objects(),
            hash_verified: report.hash_verified(),
            file_issues: report.file_issues.clone(),
            message_reports: report.messages.clone(),
        }
    }
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
        let data = vec![0u8; 32];
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
        run(&[], ValidateMode::Default, false).unwrap();
    }

    #[test]
    fn batch_json_is_valid_array() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = make_test_file(dir.path(), "a.tgm", 1);
        let p2 = make_test_file(dir.path(), "b.tgm", 1);
        run(&[p1, p2], ValidateMode::Default, true).unwrap();
    }
}
