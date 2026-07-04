// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! GRIB round-trip: `convert-grib` → `to-grib` → compare the data in the GRIB
//! domain.  Milestone 1 asserts the *value* round-trip is tight for the
//! lossless packings (`grid_simple`, `grid_ieee`); full-metadata (`grib_compare`
//! on all keys) is a later milestone once local sections are reconstructed.

use std::path::PathBuf;
use std::process::Command;

use tensogram_grib::{convert_grib_file, to_grib, ConvertOptions};

fn testdata(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("testdata");
    p.push(name);
    p
}

/// Decode object 0 of a Tensogram message into `f64` values (GRIB imports are
/// float64, little-endian).
fn f64_values(msg: &[u8]) -> Vec<f64> {
    let (_, objs) = tensogram::decode(msg, &tensogram::DecodeOptions::default()).expect("decode");
    let (desc, payload) = &objs[0];
    assert_eq!(desc.dtype, tensogram::Dtype::Float64);
    payload
        .chunks_exact(8)
        .map(|c| {
            let mut b = [0u8; 8];
            b.copy_from_slice(c);
            f64::from_le_bytes(b)
        })
        .collect()
}

fn roundtrip(name: &str, max_abs_tol: f64) {
    let src = testdata(name);
    let msgs = convert_grib_file(&src, &ConvertOptions::default()).expect("convert src");
    assert_eq!(msgs.len(), 1, "MergeAll should produce one message");
    let src_vals = f64_values(&msgs[0]);

    let grib = to_grib(&msgs[0]).expect("to_grib");
    let tmp = std::env::temp_dir().join(format!("tensogram_rt_{name}"));
    std::fs::write(&tmp, &grib).expect("write reconstructed grib");

    // Re-convert the reconstructed GRIB and compare values in the tensor domain.
    let msgs2 = convert_grib_file(&tmp, &ConvertOptions::default()).expect("convert reconstructed");
    let rt_vals = f64_values(&msgs2[0]);
    assert_eq!(
        src_vals.len(),
        rt_vals.len(),
        "{name}: value count mismatch ({} vs {})",
        src_vals.len(),
        rt_vals.len()
    );

    let (mut mae, mut mre) = (0f64, 0f64);
    for (a, b) in src_vals.iter().zip(&rt_vals) {
        let d = (a - b).abs();
        mae = mae.max(d);
        if *a != 0.0 {
            mre = mre.max(d / a.abs());
        }
    }
    eprintln!(
        "{name}: n={} max_abs={mae:.3e} max_rel={mre:.3e}",
        src_vals.len()
    );

    // Diagnostic: grib_compare on the data key, if the tool is on PATH.
    if Command::new("grib_compare").arg("-V").output().is_ok() {
        let out = Command::new("grib_compare")
            .args(["-c", "values", "-P"])
            .arg(&src)
            .arg(&tmp)
            .output()
            .expect("run grib_compare");
        eprintln!(
            "  grib_compare -c values: status={:?} {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stdout).trim()
        );
    }
    let _ = std::fs::remove_file(&tmp);

    assert!(
        mae <= max_abs_tol,
        "{name}: value round-trip max abs error {mae} exceeds {max_abs_tol}"
    );
}

#[test]
fn roundtrip_grid_simple() {
    // 12-bit simple packing: re-pack at the same bits is ~lossless.
    roundtrip("2t_simple.grib2", 1e-2);
}

#[test]
fn roundtrip_grid_ieee() {
    // IEEE storage of already-quantized values: lossless.
    roundtrip("2t_ieee.grib2", 1e-2);
}

#[test]
fn roundtrip_grid_ccsds() {
    // CCSDS/AEC (the real ECMWF packing) — re-pack at the same bits is lossless.
    roundtrip("2t.grib2", 1e-2);
}
