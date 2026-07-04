// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! NetCDF round-trip: `convert-netcdf` → `to-netcdf` → re-convert, comparing
//! the data + structure in the tensor domain.  Milestone 1 asserts variable
//! dtype, shape, and payload bytes survive the round-trip for classic and nc4
//! files (no attributes / CF packing yet).

use std::collections::BTreeMap;
use std::path::PathBuf;

use ciborium::Value as CborValue;
use tensogram_netcdf::{convert_netcdf_file, to_netcdf, ConvertOptions};

fn testdata(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("testdata");
    p.push(name);
    p
}

/// Decode a Tensogram message into `{ variable_name → (dtype, shape, payload) }`
/// so comparison is independent of variable ordering.
fn named_objects(msg: &[u8]) -> BTreeMap<String, (tensogram::Dtype, Vec<u64>, Vec<u8>)> {
    let (meta, objs) =
        tensogram::decode(msg, &tensogram::DecodeOptions::default()).expect("decode");
    let mut m = BTreeMap::new();
    for (i, (desc, payload)) in objs.iter().enumerate() {
        let name = match meta.base.get(i).and_then(|e| e.get("name")) {
            Some(CborValue::Text(s)) => s.clone(),
            _ => format!("obj{i}"),
        };
        m.insert(name, (desc.dtype, desc.shape.clone(), payload.clone()));
    }
    m
}

fn roundtrip(name: &str) {
    let src = testdata(name);
    let msgs = convert_netcdf_file(&src, &ConvertOptions::default()).expect("convert src");
    assert_eq!(
        msgs.len(),
        1,
        "{name}: SplitBy::File should produce one message"
    );

    let out = std::env::temp_dir().join(format!("tensogram_ncrt_{name}"));
    let _ = std::fs::remove_file(&out);
    to_netcdf(&msgs[0], &out).expect("to_netcdf");

    let msgs2 =
        convert_netcdf_file(&out, &ConvertOptions::default()).expect("convert reconstructed");
    assert_eq!(msgs2.len(), 1);

    let a = named_objects(&msgs[0]);
    let b = named_objects(&msgs2[0]);
    let _ = std::fs::remove_file(&out);

    assert_eq!(
        a.keys().collect::<Vec<_>>(),
        b.keys().collect::<Vec<_>>(),
        "{name}: variable set differs"
    );
    for (var, (dt_a, sh_a, pl_a)) in &a {
        let (dt_b, sh_b, pl_b) = &b[var];
        assert_eq!(dt_a, dt_b, "{name}/{var}: dtype");
        assert_eq!(sh_a, sh_b, "{name}/{var}: shape");
        assert_eq!(pl_a, pl_b, "{name}/{var}: payload bytes");
    }
    eprintln!("{name}: {} variable(s) round-tripped", a.len());
}

#[test]
fn roundtrip_simple_2d() {
    roundtrip("simple_2d.nc");
}

#[test]
fn roundtrip_nc3_classic() {
    roundtrip("nc3_classic.nc");
}

// TODO(milestone follow-up): `multi_dtype.nc` exercises i8..f64 native dtypes
// but contains a NaN, which the default encoder rejects (`allow_nan`). Enable a
// NaN-aware round-trip (tensogram NaN masks) then restore this case.
