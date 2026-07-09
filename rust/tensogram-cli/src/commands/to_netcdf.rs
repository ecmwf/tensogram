// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! `tensogram to-netcdf` — reconstruct a NetCDF file from a Tensogram file
//! (reverse of `convert-netcdf`). libnetcdf writes to a path, so `--output`
//! is required. Every message in the `.tgm` is reassembled into the one output
//! file: a file-split `.tgm` (the default) is a single message, and a
//! variable-split `.tgm` contributes one variable per message. The write is
//! atomic (temp + rename). See `plans/GRIB_NETCDF_ROUNDTRIP.md`.

use std::path::Path;

use tensogram_netcdf::to_netcdf_messages;

pub fn run(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = tensogram::TensogramFile::open(input)?;
    let count = file.message_count()?;
    if count == 0 {
        return Err(format!("{input}: no Tensogram messages").into());
    }

    let messages: Vec<Vec<u8>> = (0..count)
        .map(|i| file.read_message(i))
        .collect::<Result<_, _>>()?;
    let refs: Vec<&[u8]> = messages.iter().map(Vec::as_slice).collect();

    // `to_netcdf_messages` writes atomically (temp file + rename), so a failure
    // never leaves a partial `.nc` and never clobbers a pre-existing `output`.
    to_netcdf_messages(&refs, Path::new(output))?;
    println!("Wrote NetCDF to {output}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nc_testdata(name: &str) -> std::path::PathBuf {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../tensogram-netcdf/testdata");
        path.push(name);
        path
    }

    #[test]
    fn to_netcdf_reconstructs_file() {
        let dir = tempfile::tempdir().unwrap();
        let tgm = dir.path().join("mid.tgm");
        let msgs = tensogram_netcdf::convert_netcdf_file(
            &nc_testdata("simple_2d.nc"),
            &tensogram_netcdf::ConvertOptions::default(),
        )
        .unwrap();
        std::fs::write(&tgm, msgs.concat()).unwrap();

        let out = dir.path().join("out.nc");
        run(tgm.to_str().unwrap(), out.to_str().unwrap()).unwrap();

        // Reconstructed NetCDF must re-import cleanly.
        let back = tensogram_netcdf::convert_netcdf_file(
            &out,
            &tensogram_netcdf::ConvertOptions::default(),
        )
        .expect("reconstructed NetCDF must re-import");
        assert_eq!(back.len(), 1);
    }

    #[test]
    fn to_netcdf_missing_input_errors() {
        assert!(run("/nonexistent.tgm", "/tmp/out.nc").is_err());
    }

    #[test]
    fn to_netcdf_failure_leaves_no_output() {
        use std::collections::BTreeMap;
        use tensogram::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata};

        // A valid single-message .tgm whose object carries no NetCDF structural
        // metadata: the exporter decodes it, then fails at the missing `_file`
        // registry.  Because the write is atomic (temp + rename), no output file
        // is ever produced at `out`.
        let desc = DataObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 1,
            shape: vec![1],
            strides: vec![1],
            dtype: tensogram::Dtype::Float64,
            byte_order: ByteOrder::Little,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            masks: None,
            params: BTreeMap::new(),
        };
        let meta = GlobalMetadata {
            base: vec![BTreeMap::new()],
            ..Default::default()
        };
        let payload = 0.0_f64.to_le_bytes();
        let encoded = tensogram::encode(&meta, &[(&desc, &payload)], &Default::default())
            .expect("encode minimal message");

        let dir = tempfile::tempdir().unwrap();
        let tgm = dir.path().join("no_netcdf.tgm");
        std::fs::write(&tgm, &encoded).unwrap();
        let out = dir.path().join("out.nc");

        let result = run(tgm.to_str().unwrap(), out.to_str().unwrap());
        assert!(result.is_err(), "to-netcdf must reject a non-NetCDF .tgm");
        assert!(
            !out.exists(),
            "a failed to-netcdf must not leave a partial output file"
        );
    }
}
