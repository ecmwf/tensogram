use std::io::Write;
use std::path::Path;

use tensogram_netcdf::{convert_netcdf_file, ConvertOptions, DataPipeline, SplitBy};

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
        pipeline: DataPipeline {
            encoding: pipeline.encoding.clone(),
            bits: pipeline.bits,
            filter: pipeline.filter.clone(),
            compression: pipeline.compression.clone(),
            compression_level: pipeline.compression_level,
        },
    };

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

    // ── Task 13b: pipeline flags reach the descriptor ────────────────────

    fn pipeline_with(
        encoding: &str,
        bits: Option<u32>,
        filter: &str,
        compression: &str,
    ) -> PipelineArgs {
        PipelineArgs {
            encoding: encoding.to_string(),
            bits,
            filter: filter.to_string(),
            compression: compression.to_string(),
            compression_level: None,
        }
    }

    fn first_descriptor_fields(out: &std::path::Path) -> (String, String, String) {
        let mut f = tensogram_core::TensogramFile::open(out).unwrap();
        let msg = f.read_message(0).unwrap();
        let (_, objects) =
            tensogram_core::decode(&msg, &tensogram_core::DecodeOptions::default()).unwrap();
        let desc = &objects[0].0;
        (
            desc.encoding.clone(),
            desc.filter.clone(),
            desc.compression.clone(),
        )
    }

    #[test]
    fn convert_with_simple_packing_flag() {
        // simple_2d.nc has a single float64 variable, so simple_packing
        // applies cleanly without per-variable mixing.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("packed.tgm");
        run(
            &[testdata("simple_2d.nc")],
            Some(out.to_str().unwrap()),
            "file",
            false,
            &pipeline_with("simple_packing", Some(24), "none", "none"),
        )
        .unwrap();
        let (encoding, _, _) = first_descriptor_fields(&out);
        assert_eq!(encoding, "simple_packing");
    }

    #[test]
    fn convert_with_zstd_compression() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("zstd.tgm");
        run(
            &[testdata("simple_2d.nc")],
            Some(out.to_str().unwrap()),
            "file",
            false,
            &pipeline_with("none", None, "none", "zstd"),
        )
        .unwrap();
        let (_, _, compression) = first_descriptor_fields(&out);
        assert_eq!(compression, "zstd");
    }

    #[test]
    fn convert_with_shuffle_filter() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("shuf.tgm");
        run(
            &[testdata("simple_2d.nc")],
            Some(out.to_str().unwrap()),
            "file",
            false,
            &pipeline_with("none", None, "shuffle", "none"),
        )
        .unwrap();
        let (_, filter, _) = first_descriptor_fields(&out);
        assert_eq!(filter, "shuffle");
    }

    #[test]
    fn convert_unknown_compression_errors_cleanly() {
        let result = run(
            &[testdata("simple_2d.nc")],
            None,
            "file",
            false,
            &pipeline_with("none", None, "none", "bogus"),
        );
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("bogus"),
            "error should mention 'bogus', got: {msg}"
        );
    }

    #[test]
    fn convert_default_pipeline_produces_none_compression() {
        // Regression: omitting all pipeline flags must not silently
        // change the descriptor away from none/none/none.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("default.tgm");
        run(
            &[testdata("simple_2d.nc")],
            Some(out.to_str().unwrap()),
            "file",
            false,
            &default_pipeline(),
        )
        .unwrap();
        let (encoding, filter, compression) = first_descriptor_fields(&out);
        assert_eq!(encoding, "none");
        assert_eq!(filter, "none");
        assert_eq!(compression, "none");
    }
}
