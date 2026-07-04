// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! `tensogram to-netcdf` — reconstruct a NetCDF file from a Tensogram file
//! (reverse of `convert-netcdf`). libnetcdf writes to a path, so `--output`
//! is required. A single-message `.tgm` (the `convert-netcdf` file-split
//! default) maps to one NetCDF file. See `plans/GRIB_NETCDF_ROUNDTRIP.md`.

use std::path::Path;

use tensogram_netcdf::to_netcdf;

pub fn run(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = tensogram::TensogramFile::open(input)?;
    let count = file.message_count()?;
    match count {
        0 => return Err(format!("{input}: no Tensogram messages").into()),
        1 => {}
        n => {
            return Err(format!(
                "to-netcdf expects a single-message .tgm (convert-netcdf file-split); \
                 got {n} messages. Multi-file output is a follow-up."
            )
            .into());
        }
    }

    let msg = file.read_message(0)?;
    to_netcdf(&msg, Path::new(output))?;
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
}
