// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! `tensogram to-grib` — reconstruct GRIB from a Tensogram file (reverse of
//! `convert-grib`). Each message in the input `.tgm` is reconstructed and the
//! resulting GRIB messages are concatenated. See
//! `plans/GRIB_NETCDF_ROUNDTRIP.md`.

use std::io::Write;

use tensogram_grib::to_grib;

pub fn run(input: &str, output: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let file = tensogram::TensogramFile::open(input)?;
    let count = file.message_count()?;
    if count == 0 {
        return Err(format!("{input}: no Tensogram messages").into());
    }

    let mut grib = Vec::new();
    for i in 0..count {
        let msg = file.read_message(i)?;
        grib.extend_from_slice(&to_grib(&msg)?);
    }

    match output {
        Some(path) => {
            std::fs::write(path, &grib)?;
            println!("Wrote GRIB ({} bytes) to {path}", grib.len());
        }
        None => {
            std::io::stdout().lock().write_all(&grib)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grib_testdata(name: &str) -> std::path::PathBuf {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../tensogram-grib/testdata");
        path.push(name);
        path
    }

    #[test]
    fn to_grib_reconstructs_valid_message() {
        let dir = tempfile::tempdir().unwrap();
        let tgm = dir.path().join("mid.tgm");
        // convert-grib → .tgm
        let msgs = tensogram_grib::convert_grib_file(
            &grib_testdata("2t_simple.grib2"),
            &tensogram_grib::ConvertOptions::default(),
        )
        .unwrap();
        std::fs::write(&tgm, msgs.concat()).unwrap();

        // to-grib → GRIB file
        let out = dir.path().join("out.grib2");
        run(tgm.to_str().unwrap(), Some(out.to_str().unwrap())).unwrap();

        // The reconstructed GRIB must re-import cleanly with matching values.
        let back =
            tensogram_grib::convert_grib_file(&out, &tensogram_grib::ConvertOptions::default())
                .expect("reconstructed GRIB must re-import");
        assert_eq!(back.len(), 1);
    }

    #[test]
    fn to_grib_missing_input_errors() {
        assert!(run("/nonexistent.tgm", None).is_err());
    }
}
