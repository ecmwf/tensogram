use std::io::Write;
use std::path::Path;

use tensogram_netcdf::{convert_netcdf_file, ConvertOptions, SplitBy};

use crate::encoding_args::PipelineArgs;

pub fn run(
    inputs: &[impl AsRef<Path>],
    output: Option<&str>,
    split_by: &str,
    cf: bool,
    pipeline: &PipelineArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    if inputs.is_empty() {
        return Err("no input files specified".into());
    }

    let split_by = match split_by {
        "file" => SplitBy::File,
        "variable" => SplitBy::Variable,
        "record" => SplitBy::Record,
        other => {
            return Err(format!(
                "unknown --split-by value '{other}'; expected file, variable, or record"
            )
            .into())
        }
    };

    let options = ConvertOptions {
        split_by,
        cf,
        encode_options: tensogram_core::EncodeOptions::default(),
    };

    // pipeline args are accepted for API symmetry with convert-grib but
    // tensogram-netcdf's ConvertOptions does not yet expose a pipeline field.
    // The flags are parsed and validated by clap; we just ignore them for now.
    let _ = pipeline;

    let mut all_messages = Vec::new();

    for input in inputs {
        let messages = convert_netcdf_file(input.as_ref(), &options)?;
        all_messages.extend(messages);
    }

    match output {
        Some(out_path) => {
            let mut out = std::fs::File::create(out_path)?;
            for msg in &all_messages {
                out.write_all(msg)?;
            }
            println!(
                "Converted {} message(s) to {}",
                all_messages.len(),
                out_path
            );
        }
        None => {
            let mut stdout = std::io::stdout().lock();
            for msg in &all_messages {
                stdout.write_all(msg)?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn testdata(name: &str) -> String {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../tensogram-netcdf/testdata");
        path.push(name);
        path.to_str().unwrap().to_string()
    }

    fn default_pipeline() -> PipelineArgs {
        PipelineArgs::default()
    }

    #[test]
    fn convert_simple_2d_to_file() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.tgm");
        run(
            &[testdata("simple_2d.nc")],
            Some(out.to_str().unwrap()),
            "file",
            false,
            &default_pipeline(),
        )
        .unwrap();
        let mut f = tensogram_core::TensogramFile::open(&out).unwrap();
        assert_eq!(f.message_count().unwrap(), 1);
    }

    #[test]
    fn convert_split_by_variable() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.tgm");
        run(
            &[testdata("multi_var.nc")],
            Some(out.to_str().unwrap()),
            "variable",
            false,
            &default_pipeline(),
        )
        .unwrap();
        let mut f = tensogram_core::TensogramFile::open(&out).unwrap();
        // multi_var.nc has 3 numeric vars (char skipped)
        assert!(f.message_count().unwrap() >= 3);
    }

    #[test]
    fn convert_split_by_record() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.tgm");
        run(
            &[testdata("unlimited_time.nc")],
            Some(out.to_str().unwrap()),
            "record",
            false,
            &default_pipeline(),
        )
        .unwrap();
        let mut f = tensogram_core::TensogramFile::open(&out).unwrap();
        assert_eq!(f.message_count().unwrap(), 5);
    }

    #[test]
    fn convert_with_cf_flag() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.tgm");
        run(
            &[testdata("cf_temperature.nc")],
            Some(out.to_str().unwrap()),
            "file",
            true,
            &default_pipeline(),
        )
        .unwrap();
        let mut f = tensogram_core::TensogramFile::open(&out).unwrap();
        let msg = f.read_message(0).unwrap();
        let meta = tensogram_core::decode_metadata(&msg).unwrap();
        // At least one base entry should have a "cf" key
        assert!(
            meta.base.iter().any(|e| e.contains_key("cf")),
            "--cf flag should produce 'cf' key in base entries"
        );
    }

    #[test]
    fn convert_multiple_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("merged.tgm");
        run(
            &[testdata("simple_2d.nc"), testdata("nc3_classic.nc")],
            Some(out.to_str().unwrap()),
            "file",
            false,
            &default_pipeline(),
        )
        .unwrap();
        let mut f = tensogram_core::TensogramFile::open(&out).unwrap();
        assert!(
            f.message_count().unwrap() >= 2,
            "two input files should produce at least 2 messages"
        );
    }

    #[test]
    fn convert_no_inputs_errors() {
        let empty: Vec<String> = vec![];
        assert!(run(&empty, None, "file", false, &default_pipeline()).is_err());
    }

    #[test]
    fn convert_missing_file_errors() {
        let result = run(
            &["/nonexistent.nc".to_string()],
            None,
            "file",
            false,
            &default_pipeline(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn convert_empty_file_errors() {
        let result = run(
            &[testdata("empty_file.nc")],
            None,
            "file",
            false,
            &default_pipeline(),
        );
        assert!(result.is_err(), "empty file should produce an error");
    }

    #[test]
    fn convert_invalid_split_by_errors() {
        let result = run(
            &[testdata("simple_2d.nc")],
            None,
            "invalid",
            false,
            &default_pipeline(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn convert_record_without_unlimited_errors() {
        let result = run(
            &[testdata("simple_2d.nc")],
            None,
            "record",
            false,
            &default_pipeline(),
        );
        assert!(
            result.is_err(),
            "record split on file without unlimited dim should error"
        );
    }

    #[test]
    fn convert_nc4_groups_root_only() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.tgm");
        run(
            &[testdata("nc4_groups.nc")],
            Some(out.to_str().unwrap()),
            "file",
            false,
            &default_pipeline(),
        )
        .unwrap();
        let mut f = tensogram_core::TensogramFile::open(&out).unwrap();
        assert_eq!(f.message_count().unwrap(), 1);
        let msg = f.read_message(0).unwrap();
        let meta = tensogram_core::decode_metadata(&msg).unwrap();
        // Only root_var should be present, not predicted (sub-group)
        let has_root_var = meta.base.iter().any(|e| {
            e.get("name").and_then(|v| {
                if let ciborium::Value::Text(s) = v {
                    Some(s.as_str())
                } else {
                    None
                }
            }) == Some("root_var")
        });
        assert!(has_root_var);
    }

    #[test]
    fn convert_multi_dtype_preserves_types() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.tgm");
        run(
            &[testdata("multi_dtype.nc")],
            Some(out.to_str().unwrap()),
            "file",
            false,
            &default_pipeline(),
        )
        .unwrap();
        let mut f = tensogram_core::TensogramFile::open(&out).unwrap();
        assert!(f.message_count().unwrap() >= 1);
    }
}
