// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::io::Write;
use std::path::Path;

use tensogram_grib::{ConvertOptions, DataPipeline, Grouping, convert_grib_file};

use crate::encoding_args::PipelineArgs;

#[allow(clippy::too_many_arguments)]
pub fn run(
    inputs: &[impl AsRef<Path>],
    output: Option<&str>,
    split: bool,
    all_keys: bool,
    pipeline: &PipelineArgs,
    threads: u32,
    reject_nan: bool,
    reject_inf: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if inputs.is_empty() {
        return Err("no input files specified".into());
    }

    let grouping = if split {
        Grouping::OneToOne
    } else {
        Grouping::MergeAll
    };

    let options = ConvertOptions {
        grouping,
        preserve_all_keys: all_keys,
        pipeline: DataPipeline {
            encoding: pipeline.encoding.clone(),
            bits: pipeline.bits,
            filter: pipeline.filter.clone(),
            compression: pipeline.compression.clone(),
            compression_level: pipeline.compression_level,
        },
        encode_options: tensogram::EncodeOptions {
            threads,
            reject_nan,
            reject_inf,
            ..Default::default()
        },
    };

    let mut all_messages = Vec::new();

    for input in inputs {
        let messages = convert_grib_file(input.as_ref(), &options)?;
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
        path.push("../tensogram-grib/testdata");
        path.push(name);
        path.to_str().unwrap().to_string()
    }

    fn default_pipeline() -> PipelineArgs {
        PipelineArgs::default()
    }

    #[test]
    fn convert_single_grib_merge() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.tgm");
        run(
            &[testdata("2t.grib2")],
            Some(out.to_str().unwrap()),
            false,
            false,
            &default_pipeline(),
            0,
            false,
            false,
        )
        .unwrap();
        let f = tensogram::TensogramFile::open(&out).unwrap();
        assert!(f.message_count().unwrap() >= 1);
    }

    #[test]
    fn convert_single_grib_split() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.tgm");
        run(
            &[testdata("2t.grib2")],
            Some(out.to_str().unwrap()),
            true,
            false,
            &default_pipeline(),
            0,
            false,
            false,
        )
        .unwrap();
        let f = tensogram::TensogramFile::open(&out).unwrap();
        assert!(f.message_count().unwrap() >= 1);
    }

    #[test]
    fn convert_with_all_keys() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.tgm");
        run(
            &[testdata("2t.grib2")],
            Some(out.to_str().unwrap()),
            false,
            true,
            &default_pipeline(),
            0,
            false,
            false,
        )
        .unwrap();
        let f = tensogram::TensogramFile::open(&out).unwrap();
        let msg = f.read_message(0).unwrap();
        let meta = tensogram::decode_metadata(&msg).unwrap();
        assert!(
            meta.base.iter().any(|entry| entry.contains_key("grib")),
            "all_keys should produce grib namespace in base entries"
        );
    }

    #[test]
    fn convert_multiple_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("merged.tgm");
        run(
            &[testdata("2t.grib2"), testdata("lsm.grib2")],
            Some(out.to_str().unwrap()),
            false,
            false,
            &default_pipeline(),
            0,
            false,
            false,
        )
        .unwrap();
        let f = tensogram::TensogramFile::open(&out).unwrap();
        assert!(
            f.message_count().unwrap() >= 2,
            "two input files should produce at least 2 messages"
        );
    }

    #[test]
    fn convert_no_inputs_errors() {
        let empty: Vec<String> = vec![];
        assert!(run(&empty, None, false, false, &default_pipeline(), 0,
 false,
 false,).is_err());
    }

    #[test]
    fn convert_missing_file_errors() {
        let result = run(
            &["/nonexistent.grib2".to_string()],
            None,
            false,
            false,
            &default_pipeline(),
            0,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn convert_pressure_level_merge() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("pl.tgm");
        run(
            &[testdata("q_150.grib2")],
            Some(out.to_str().unwrap()),
            false,
            false,
            &default_pipeline(),
            0,
            false,
            false,
        )
        .unwrap();
        let f = tensogram::TensogramFile::open(&out).unwrap();
        assert!(f.message_count().unwrap() >= 1);
    }

    #[test]
    fn convert_split_with_all_keys() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.tgm");
        run(
            &[testdata("2t.grib2")],
            Some(out.to_str().unwrap()),
            true,
            true,
            &default_pipeline(),
            0,
            false,
            false,
        )
        .unwrap();
        let f = tensogram::TensogramFile::open(&out).unwrap();
        let msg = f.read_message(0).unwrap();
        let meta = tensogram::decode_metadata(&msg).unwrap();
        assert!(meta.base.iter().any(|e| e.contains_key("grib")));
        assert!(meta.base.iter().any(|e| e.contains_key("mars")));
    }

    #[test]
    fn convert_round_trip_data_integrity() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("rt.tgm");
        run(
            &[testdata("t_600.grib2")],
            Some(out.to_str().unwrap()),
            true,
            false,
            &default_pipeline(),
            0,
            false,
            false,
        )
        .unwrap();
        let f = tensogram::TensogramFile::open(&out).unwrap();
        let msg = f.read_message(0).unwrap();
        let (_, objects) = tensogram::decode(&msg, &tensogram::DecodeOptions::default()).unwrap();
        let (desc, data) = &objects[0];
        assert_eq!(desc.dtype, tensogram::Dtype::Float64);
        assert!(!data.is_empty());
        assert!(!desc.shape.is_empty());
    }
}
