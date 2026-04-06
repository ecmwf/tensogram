use std::io::Write;
use std::path::Path;

use tensogram_grib::{convert_grib_file, ConvertOptions, Grouping};

/// Convert GRIB messages to Tensogram format.
pub fn run(
    inputs: &[impl AsRef<Path>],
    output: Option<&str>,
    split: bool,
    all_keys: bool,
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
        ..Default::default()
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

    #[test]
    fn convert_single_grib_merge() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.tgm");
        run(
            &[testdata("2t.grib2")],
            Some(out.to_str().unwrap()),
            false, // merge mode
            false, // no all_keys
        )
        .unwrap();
        // Verify output is valid tensogram
        let mut f = tensogram_core::TensogramFile::open(&out).unwrap();
        assert!(f.message_count().unwrap() >= 1);
    }

    #[test]
    fn convert_single_grib_split() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.tgm");
        run(
            &[testdata("2t.grib2")],
            Some(out.to_str().unwrap()),
            true, // split mode
            false,
        )
        .unwrap();
        let mut f = tensogram_core::TensogramFile::open(&out).unwrap();
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
            true, // all_keys enabled
        )
        .unwrap();
        // Verify grib namespace keys are in metadata
        let mut f = tensogram_core::TensogramFile::open(&out).unwrap();
        let msg = f.read_message(0).unwrap();
        let meta = tensogram_core::decode_metadata(&msg).unwrap();
        // With all_keys, common should contain "grib" key
        assert!(
            meta.common.contains_key("grib"),
            "all_keys should produce grib namespace in common"
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
        )
        .unwrap();
        // Each input GRIB file → 1 merged message, so 2 files → 2 messages
        let mut f = tensogram_core::TensogramFile::open(&out).unwrap();
        assert!(
            f.message_count().unwrap() >= 2,
            "two input files should produce at least 2 messages"
        );
    }

    // Note: the stdout branch (output=None) is not unit-tested here because
    // it writes binary Tensogram bytes to stdout, which bloats captured test
    // output and makes failures noisy. The branch is trivial (write_all to
    // stdout lock) and is exercised by the existing CI integration tests.

    #[test]
    fn convert_no_inputs_errors() {
        let empty: Vec<String> = vec![];
        assert!(run(&empty, None, false, false).is_err());
    }

    #[test]
    fn convert_missing_file_errors() {
        let result = run(&["/nonexistent.grib2".to_string()], None, false, false);
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
        )
        .unwrap();
        let mut f = tensogram_core::TensogramFile::open(&out).unwrap();
        assert!(f.message_count().unwrap() >= 1);
    }

    #[test]
    fn convert_split_with_all_keys() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.tgm");
        run(
            &[testdata("2t.grib2")],
            Some(out.to_str().unwrap()),
            true, // split
            true, // all_keys
        )
        .unwrap();
        let mut f = tensogram_core::TensogramFile::open(&out).unwrap();
        let msg = f.read_message(0).unwrap();
        let meta = tensogram_core::decode_metadata(&msg).unwrap();
        assert!(meta.common.contains_key("grib"));
        assert!(meta.common.contains_key("mars"));
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
        )
        .unwrap();
        // Decode and verify data is f64 array with correct shape
        let mut f = tensogram_core::TensogramFile::open(&out).unwrap();
        let msg = f.read_message(0).unwrap();
        let (_, objects) =
            tensogram_core::decode(&msg, &tensogram_core::DecodeOptions::default()).unwrap();
        let (desc, data) = &objects[0];
        assert_eq!(desc.dtype, tensogram_core::Dtype::Float64);
        assert!(!data.is_empty());
        // shape should be 2D [Nj, Ni] or 1D [N]
        assert!(!desc.shape.is_empty());
    }
}
