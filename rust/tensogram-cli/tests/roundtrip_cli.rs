// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! End-to-end CLI round-trip, compared **in the source format with that
//! format's own tools** (`plans/TEST.md`, "Round-trip conversion testing"):
//!
//! ```text
//! tensogram convert-FORMAT in.FMT  -o mid.tgm
//! tensogram to-FORMAT      mid.tgm -o out.FMT
//! <native compare in.FMT vs out.FMT>
//! ```
//!
//! Gated on the `grib` / `netcdf` cargo features and on the presence of the
//! native comparison tools (`grib_compare`, `ncdump`); skipped, not failed,
//! when a tool is absent.

#![cfg(any(feature = "grib", feature = "netcdf"))]

use std::path::{Path, PathBuf};
use std::process::Command;

/// The freshly-built `tensogram` binary (with the features under test).
fn cli() -> &'static str {
    env!("CARGO_BIN_EXE_tensogram")
}

/// True if `tool` can be launched (i.e. is on PATH).
fn tool_present(tool: &str) -> bool {
    Command::new(tool).arg("--help").output().is_ok()
}

fn run_ok(args: &[&str]) {
    let status = Command::new(cli())
        .args(args)
        .status()
        .unwrap_or_else(|e| panic!("failed to launch {}: {e}", cli()));
    assert!(status.success(), "CLI failed: tensogram {}", args.join(" "));
}

#[cfg(feature = "grib")]
mod grib {
    use super::*;

    fn fixture(name: &str) -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("../tensogram-grib/testdata");
        p.push(name);
        p
    }

    fn roundtrip(name: &str) {
        if !tool_present("grib_compare") {
            eprintln!("skip cli_grib_roundtrip({name}): grib_compare not on PATH");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let src = fixture(name);
        let tgm = dir.path().join("mid.tgm");
        let out = dir.path().join("out.grib2");

        run_ok(&[
            "convert-grib",
            src.to_str().unwrap(),
            "-o",
            tgm.to_str().unwrap(),
        ]);
        run_ok(&[
            "to-grib",
            tgm.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
        ]);

        // Compare the data in the GRIB domain with ecCodes' own tool.
        let cmp = Command::new("grib_compare")
            .args(["-c", "values"])
            .arg(&src)
            .arg(&out)
            .output()
            .expect("run grib_compare");
        assert!(
            cmp.status.success(),
            "grib_compare -c values mismatch for {name}:\n{}\n{}",
            String::from_utf8_lossy(&cmp.stdout),
            String::from_utf8_lossy(&cmp.stderr),
        );
    }

    #[test]
    fn cli_grib_roundtrip_simple() {
        roundtrip("2t_simple.grib2");
    }

    #[test]
    fn cli_grib_roundtrip_ieee() {
        roundtrip("2t_ieee.grib2");
    }
}

#[cfg(feature = "netcdf")]
mod netcdf {
    use super::*;

    fn fixture(name: &str) -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("../tensogram-netcdf/testdata");
        p.push(name);
        p
    }

    /// `ncdump` output with the leading `netcdf <name> {` line normalised so
    /// the (differing) filenames don't cause spurious diffs.
    fn ncdump_normalised(path: &Path) -> String {
        let out = Command::new("ncdump")
            .arg(path)
            .output()
            .expect("run ncdump");
        assert!(out.status.success(), "ncdump failed for {path:?}");
        let text = String::from_utf8_lossy(&out.stdout);
        text.lines()
            .map(|l| {
                if l.trim_start().starts_with("netcdf ") {
                    "netcdf X {"
                } else {
                    l
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn roundtrip(name: &str) {
        if !tool_present("ncdump") {
            eprintln!("skip cli_netcdf_roundtrip({name}): ncdump not on PATH");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let src = fixture(name);
        let tgm = dir.path().join("mid.tgm");
        let out = dir.path().join("out.nc");

        run_ok(&[
            "convert-netcdf",
            src.to_str().unwrap(),
            "-o",
            tgm.to_str().unwrap(),
        ]);
        run_ok(&[
            "to-netcdf",
            tgm.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
        ]);

        // Compare structure + data in the NetCDF domain (nccmp is not
        // installed here, so use an ncdump structural+data diff).
        assert_eq!(
            ncdump_normalised(&src),
            ncdump_normalised(&out),
            "ncdump round-trip mismatch for {name}"
        );
    }

    #[test]
    fn cli_netcdf_roundtrip_simple_2d() {
        roundtrip("simple_2d.nc");
    }

    #[test]
    fn cli_netcdf_roundtrip_nc3_classic() {
        roundtrip("nc3_classic.nc");
    }

    /// Every native dtype + a scalar + a NaN-bearing variable: `ncdump` output
    /// (structure, data, and the `NaN` element) must be identical after the
    /// round-trip, exercising the `allow_nan` mask path end-to-end.
    #[test]
    fn cli_netcdf_roundtrip_multi_dtype() {
        roundtrip("multi_dtype.nc");
    }
}
