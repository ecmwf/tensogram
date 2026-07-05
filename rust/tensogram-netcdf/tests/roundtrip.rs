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

/// Read every attribute of variable `var` as `(name, "{:?}")` pairs, sorted.
/// The `Debug` string captures both the exact `AttributeValue` *type* and the
/// value, so comparing two files' outputs proves both survived the round-trip.
fn var_attr_debug(path: &std::path::Path, var: &str) -> Vec<(String, String)> {
    let f = netcdf::open(path).expect("open");
    let v = f.variable(var).expect("variable present");
    let mut out: Vec<(String, String)> = v
        .attributes()
        .map(|a| {
            (
                a.name().to_string(),
                format!("{:?}", a.value().expect("attr value")),
            )
        })
        .collect();
    out.sort();
    out
}

/// Same as [`var_attr_debug`] for file-level (global) attributes.
fn global_attr_debug(path: &std::path::Path) -> Vec<(String, String)> {
    let f = netcdf::open(path).expect("open");
    let mut out: Vec<(String, String)> = f
        .attributes()
        .map(|a| {
            (
                a.name().to_string(),
                format!("{:?}", a.value().expect("attr value")),
            )
        })
        .collect();
    out.sort();
    out
}

/// Exact attribute-*type* round-trip: a variable carrying float / double / int /
/// short / byte / uint / string / int-array / float-array attributes (plus a
/// float global attribute) must come back with byte-for-byte identical
/// `AttributeValue` variants — not widened to `double` / `int64`.
///
/// The fixture is built in-process (no `scale_factor` / `_FillValue`, so the
/// variable keeps its native dtype and no NaN unpacking occurs).
#[test]
fn roundtrip_attr_types_exact() {
    let dir = std::env::temp_dir().join("tensogram_attr_types_exact");
    std::fs::create_dir_all(&dir).expect("mkdir");
    let src = dir.join("src.nc");
    let out = dir.join("out.nc");
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);

    {
        let mut f = netcdf::create(&src).expect("create src");
        f.add_dimension("n", 3).expect("dim");
        let mut v = f.add_variable::<i32>("v", &["n"]).expect("var");
        v.put_attribute("att_float", 1.5_f32).expect("att_float");
        v.put_attribute("att_double", 2.5_f64).expect("att_double");
        v.put_attribute("att_int", 7_i32).expect("att_int");
        v.put_attribute("att_short", 3_i16).expect("att_short");
        v.put_attribute("att_byte", -5_i8).expect("att_byte");
        v.put_attribute("att_uint", 9_u32).expect("att_uint");
        v.put_attribute("att_str", "hello").expect("att_str");
        v.put_attribute("att_ints", vec![1_i32, 2, 3])
            .expect("att_ints");
        v.put_attribute("att_floats", vec![1.0_f32, 2.0])
            .expect("att_floats");
        v.put_values(&[1_i32, 2, 3], ..).expect("values");
        f.add_attribute("g_float", 4.5_f32).expect("g_float");
    }

    let src_var = var_attr_debug(&src, "v");
    let src_glob = global_attr_debug(&src);

    let msgs = convert_netcdf_file(&src, &ConvertOptions::default()).expect("convert");
    to_netcdf(&msgs[0], &out).expect("to_netcdf");

    let out_var = var_attr_debug(&out, "v");
    let out_glob = global_attr_debug(&out);

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);

    assert_eq!(
        src_var, out_var,
        "variable attribute types/values changed across round-trip"
    );
    assert_eq!(
        src_glob, out_glob,
        "global attribute types/values changed across round-trip"
    );
}

#[test]
fn roundtrip_simple_2d() {
    roundtrip("simple_2d.nc");
}

#[test]
fn roundtrip_nc3_classic() {
    roundtrip("nc3_classic.nc");
}

/// `multi_dtype.nc` exercises every native dtype (i8..u64, f32, f64), a scalar
/// (`pi`), and a `f64_with_nan` variable holding a genuine NaN.  The converter
/// enables `allow_nan`, so the NaN is masked on encode and restored on decode;
/// the round-trip must preserve the canonical NaN bit pattern (byte-equal
/// payload) rather than reject or corrupt it.
#[test]
fn roundtrip_multi_dtype() {
    roundtrip("multi_dtype.nc");
}

/// A CF-packed variable (`short` with `scale_factor` / `add_offset`) is unpacked
/// to physical f64 on import.  The round-trip must be stable: the second
/// conversion of the reconstructed file (which reads the f64 values natively,
/// with no packing to redo) yields the identical tensor payload.
#[test]
fn roundtrip_cf_packed_unpacks() {
    roundtrip("cf_temperature_deflate.nc");
}

/// Locks in the packing-attribute drop: because `temperature` was unpacked to
/// f64, the reconstructed file must carry no `scale_factor` / `add_offset` /
/// `_FillValue` (keeping them would double-unpack and mismatch the f64 fill
/// type), while descriptive attributes like `units` survive.
#[test]
fn cf_packed_export_drops_packing_attrs() {
    let src = testdata("cf_temperature_deflate.nc");
    let msgs = convert_netcdf_file(&src, &ConvertOptions::default()).expect("convert");
    let out = std::env::temp_dir().join("tensogram_cf_drop_packing.nc");
    let _ = std::fs::remove_file(&out);
    to_netcdf(&msgs[0], &out).expect("to_netcdf must succeed for CF-packed input");

    let f = netcdf::open(&out).expect("open reconstructed");
    let v = f.variable("temperature").expect("temperature var present");
    assert!(
        v.attribute("scale_factor").is_none(),
        "scale_factor must be dropped from an unpacked variable"
    );
    assert!(
        v.attribute("add_offset").is_none(),
        "add_offset must be dropped from an unpacked variable"
    );
    assert!(
        v.attribute("_FillValue").is_none(),
        "packed _FillValue must be dropped from an unpacked variable"
    );
    assert!(
        v.attribute("units").is_some(),
        "descriptive attributes must survive unpacking"
    );
    let _ = std::fs::remove_file(&out);
}
