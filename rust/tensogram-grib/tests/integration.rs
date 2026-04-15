// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Integration tests for tensogram-grib.
//!
//! Each test uses real ECMWF opendata GRIB fixtures stored in `testdata/`.
//! Run with:  cargo test --manifest-path crates/tensogram-grib/Cargo.toml

use std::path::{Path, PathBuf};

use ciborium::Value as CborValue;
use tensogram_core::{decode, DecodeOptions};
use tensogram_grib::{convert_grib_file, ConvertOptions, Grouping};

/// Path to the testdata directory.
fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata")
}

/// Default decode options (no hash verification).
fn decode_opts() -> DecodeOptions {
    DecodeOptions::default()
}

/// Look up a string key in a `CborValue::Map`.
fn cbor_map_get<'a>(map: &'a CborValue, key: &str) -> Option<&'a CborValue> {
    if let CborValue::Map(entries) = map {
        entries.iter().find_map(|(k, v)| match k {
            CborValue::Text(s) if s == key => Some(v),
            _ => None,
        })
    } else {
        None
    }
}

/// Extract the `base[0]["mars"]` sub-map from global metadata.
///
/// For single-object messages, base[0] holds all metadata.
/// For multi-object messages with identical mars keys, we look at base[0].
fn get_mars_from_base(meta: &tensogram_core::GlobalMetadata) -> &CborValue {
    meta.base
        .first()
        .expect("base must have at least one entry")
        .get("mars")
        .expect("base[0] must contain 'mars' key")
}

// ──────────────────────────────────────────────────────────────────────
// 1. test_lsm_convert
// ──────────────────────────────────────────────────────────────────────

/// Convert lsm.grib2 (land-sea mask, surface) and verify metadata.
#[test]
fn test_lsm_convert() {
    let path = testdata().join("lsm.grib2");
    let opts = ConvertOptions::default(); // MergeAll
    let messages = convert_grib_file(&path, &opts).expect("convert lsm.grib2");

    assert_eq!(messages.len(), 1, "single GRIB -> single Tensogram message");

    let (meta, objects) = decode(&messages[0], &decode_opts()).expect("decode");
    assert_eq!(objects.len(), 1, "one data object");

    // MARS keys should be under base[0]["mars"].
    let mars = get_mars_from_base(&meta);
    assert!(cbor_map_get(mars, "class").is_some(), "missing mars.class");
    assert!(cbor_map_get(mars, "type").is_some(), "missing mars.type");
    assert!(
        cbor_map_get(mars, "stream").is_some(),
        "missing mars.stream"
    );
    assert!(
        cbor_map_get(mars, "expver").is_some(),
        "missing mars.expver"
    );

    // Verify a parameter-related key exists.
    let has_param = cbor_map_get(mars, "param").is_some()
        || cbor_map_get(mars, "shortName").is_some()
        || cbor_map_get(mars, "paramId").is_some();
    assert!(has_param, "no parameter identification key in mars");

    // gridType stored as "grid" in the mars namespace.
    assert_eq!(
        cbor_map_get(mars, "grid"),
        Some(&CborValue::Text("regular_ll".to_string())),
        "mars.grid should be 'regular_ll'"
    );

    // Shape: 0.25 deg global grid -> 721 x 1440.
    let (desc, _) = &objects[0];
    assert_eq!(desc.ndim, 2);
    assert_eq!(desc.shape, vec![721, 1440]);
}

// ──────────────────────────────────────────────────────────────────────
// 2. test_2t_round_trip
// ──────────────────────────────────────────────────────────────────────

/// Convert 2t.grib2 (2m temperature), decode, verify f64 data round-trips.
#[test]
fn test_2t_round_trip() {
    let path = testdata().join("2t.grib2");
    let opts = ConvertOptions::default();
    let messages = convert_grib_file(&path, &opts).expect("convert 2t.grib2");

    let (_, objects) = decode(&messages[0], &decode_opts()).expect("decode");
    let (desc, raw_bytes) = &objects[0];

    assert_eq!(desc.dtype, tensogram_core::Dtype::Float64);
    let expected_elements: usize = desc.shape.iter().product::<u64>() as usize;
    assert_eq!(
        raw_bytes.len(),
        expected_elements * 8,
        "decoded byte count = elements * 8"
    );

    // Spot-check: all values should be finite f64.
    let values: Vec<f64> = raw_bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!(
        values.iter().all(|v| v.is_finite()),
        "all 2t values should be finite"
    );
    // Temperature sanity: at least some values should be > 200 K.
    assert!(
        values.iter().any(|&v| v > 200.0),
        "2t values should contain temperatures > 200 K"
    );
}

// ──────────────────────────────────────────────────────────────────────
// 3. test_q_pl_convert
// ──────────────────────────────────────────────────────────────────────

/// Convert q_150.grib2 (specific humidity at 150 hPa) and verify
/// pressure-level metadata.
#[test]
fn test_q_pl_convert() {
    let path = testdata().join("q_150.grib2");
    let opts = ConvertOptions::default();
    let messages = convert_grib_file(&path, &opts).expect("convert q_150.grib2");

    let (meta, objects) = decode(&messages[0], &decode_opts()).expect("decode");
    assert_eq!(objects.len(), 1);

    // Should have pressure-level related keys under mars.
    let mars = get_mars_from_base(&meta);
    let has_level =
        cbor_map_get(mars, "levelist").is_some() || cbor_map_get(mars, "level").is_some();
    assert!(has_level, "missing level key for pressure-level data");

    // Shape check.
    let (desc, _) = &objects[0];
    assert_eq!(desc.ndim, 2);
    assert_eq!(desc.shape, vec![721, 1440]);
}

// ──────────────────────────────────────────────────────────────────────
// 4. test_t_pl_round_trip
// ──────────────────────────────────────────────────────────────────────

/// Convert t_600.grib2 (temperature at 600 hPa), decode, round-trip check.
#[test]
fn test_t_pl_round_trip() {
    let path = testdata().join("t_600.grib2");
    let opts = ConvertOptions::default();
    let messages = convert_grib_file(&path, &opts).expect("convert t_600.grib2");

    let (_, objects) = decode(&messages[0], &decode_opts()).expect("decode");
    let (desc, raw_bytes) = &objects[0];

    let expected_elements: usize = desc.shape.iter().product::<u64>() as usize;
    assert_eq!(raw_bytes.len(), expected_elements * 8);

    let values: Vec<f64> = raw_bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!(values.iter().all(|v| v.is_finite()));
    // 600 hPa temperature: should have values in a reasonable range (180-320 K).
    assert!(
        values.iter().any(|&v| v > 200.0 && v < 320.0),
        "t at 600 hPa should have values in [200, 320] K range"
    );
}

// ──────────────────────────────────────────────────────────────────────
// 5. test_multi_merge
// ──────────────────────────────────────────────────────────────────────

/// MergeAll with 2 GRIB files (lsm + 2t) concatenated -> 1 Tensogram
/// message with 2 objects. Each `base[i]` holds ALL metadata for that
/// object independently.
#[test]
fn test_multi_merge() {
    let combined = make_combined_grib(&["lsm.grib2", "2t.grib2"]);
    let opts = ConvertOptions {
        grouping: Grouping::MergeAll,
        ..ConvertOptions::default()
    };
    let messages = convert_grib_file(combined.path(), &opts).expect("convert merged");

    assert_eq!(messages.len(), 1, "MergeAll -> 1 Tensogram message");

    let (meta, objects) = decode(&messages[0], &decode_opts()).expect("decode");
    assert_eq!(objects.len(), 2, "2 GRIB messages -> 2 data objects");

    // base should have 2 entries (one per object)
    assert_eq!(meta.base.len(), 2, "base should have 2 entries");

    // Each base entry should have mars keys.
    for (i, entry) in meta.base.iter().enumerate() {
        assert!(entry.contains_key("mars"), "base[{i}] must have mars");
    }

    // Both entries should have auto-populated _reserved_.tensor.
    for (i, entry) in meta.base.iter().enumerate() {
        assert!(
            entry.contains_key("_reserved_"),
            "base[{i}] must have _reserved_"
        );
    }

    // mars.class should be present in both (shared across lsm + 2t).
    for (i, entry) in meta.base.iter().enumerate() {
        let mars = entry.get("mars").expect("mars key");
        assert!(
            cbor_map_get(mars, "class").is_some(),
            "base[{i}] mars.class should be present"
        );
    }
}

// ──────────────────────────────────────────────────────────────────────
// 6. test_multi_split
// ──────────────────────────────────────────────────────────────────────

/// OneToOne with 2 GRIB files -> 2 Tensogram messages, each with 1 object.
#[test]
fn test_multi_split() {
    let combined = make_combined_grib(&["lsm.grib2", "2t.grib2"]);
    let opts = ConvertOptions {
        grouping: Grouping::OneToOne,
        ..ConvertOptions::default()
    };
    let messages = convert_grib_file(combined.path(), &opts).expect("convert split");

    assert_eq!(messages.len(), 2, "OneToOne -> 2 Tensogram messages");

    for (i, msg_bytes) in messages.iter().enumerate() {
        let (meta, objects) = decode(msg_bytes, &decode_opts()).expect("decode");
        assert_eq!(objects.len(), 1, "message {} has 1 object", i);
        // Each message should have all MARS keys in base[0]["mars"].
        assert!(
            meta.base.first().is_some_and(|e| e.contains_key("mars")),
            "message {} base[0] must have mars",
            i
        );
    }
}

// ──────────────────────────────────────────────────────────────────────
// 7. test_payload_objects_metadata
// ──────────────────────────────────────────────────────────────────────

/// Verify base entries have _reserved_.tensor (ndim, shape, strides, dtype)
/// and mars.grid.
#[test]
fn test_base_entry_metadata() {
    let path = testdata().join("2t.grib2");
    let opts = ConvertOptions::default();
    let messages = convert_grib_file(&path, &opts).expect("convert 2t.grib2");

    let (meta, _) = decode(&messages[0], &decode_opts()).expect("decode");

    // base should have one entry.
    assert_eq!(meta.base.len(), 1, "single object -> one base entry");

    let entry = &meta.base[0];

    // _reserved_.tensor should have ndim, shape, strides, dtype.
    let reserved = entry.get("_reserved_").expect("must have _reserved_");
    let tensor = cbor_map_get(reserved, "tensor").expect("must have tensor");
    assert!(cbor_map_get(tensor, "ndim").is_some(), "must have ndim");
    assert!(cbor_map_get(tensor, "shape").is_some(), "must have shape");
    assert!(
        cbor_map_get(tensor, "strides").is_some(),
        "must have strides"
    );
    assert_eq!(
        cbor_map_get(tensor, "dtype"),
        Some(&CborValue::Text("float64".to_string())),
        "dtype should be 'float64'"
    );

    // All mars keys (including grid) are in base[0]["mars"].
    let mars = get_mars_from_base(&meta);
    assert_eq!(
        cbor_map_get(mars, "grid"),
        Some(&CborValue::Text("regular_ll".to_string())),
        "mars.grid should be 'regular_ll'"
    );
}

// ──────────────────────────────────────────────────────────────────────
// 8. test_all_keys_single_object
// ──────────────────────────────────────────────────────────────────────

/// With `preserve_all_keys`, base[0]["grib"] should contain namespace sub-maps.
#[test]
fn test_all_keys_single_object() {
    let path = testdata().join("2t.grib2");
    let opts = ConvertOptions {
        preserve_all_keys: true,
        ..ConvertOptions::default()
    };
    let messages = convert_grib_file(&path, &opts).expect("convert 2t.grib2 all-keys");

    let (meta, _) = decode(&messages[0], &decode_opts()).expect("decode");

    let entry = &meta.base[0];

    // mars keys in base[0]["mars"]
    assert!(entry.contains_key("mars"), "must have mars");

    // grib namespace keys in base[0]["grib"]
    let grib = entry
        .get("grib")
        .expect("base[0] must contain 'grib' when preserve_all_keys is on");

    // grib should be a map of namespace maps
    let grib_map = match grib {
        CborValue::Map(m) => m,
        other => panic!("grib should be a map, got: {:?}", other),
    };

    // Check that expected namespaces are present
    let ns_names: Vec<&str> = grib_map
        .iter()
        .filter_map(|(k, _)| match k {
            CborValue::Text(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();

    assert!(ns_names.contains(&"geography"), "must have geography ns");
    assert!(ns_names.contains(&"time"), "must have time ns");
    assert!(ns_names.contains(&"parameter"), "must have parameter ns");

    // Check a specific key: geography.Ni should be 1440
    let geo = cbor_map_get(grib, "geography").expect("geography ns");
    let ni = cbor_map_get(geo, "Ni");
    assert_eq!(
        ni,
        Some(&CborValue::Integer(1440.into())),
        "geography.Ni should be 1440"
    );
}

// ──────────────────────────────────────────────────────────────────────
// 9. test_all_keys_multi_merge
// ──────────────────────────────────────────────────────────────────────

/// With `preserve_all_keys` and MergeAll, each `base[i]` holds all grib
/// namespace keys for that object independently.
#[test]
fn test_all_keys_multi_merge() {
    let combined = make_combined_grib(&["lsm.grib2", "2t.grib2"]);
    let opts = ConvertOptions {
        grouping: Grouping::MergeAll,
        preserve_all_keys: true,
        ..ConvertOptions::default()
    };
    let messages = convert_grib_file(combined.path(), &opts).expect("convert all-keys merged");

    let (meta, objects) = decode(&messages[0], &decode_opts()).expect("decode");
    assert_eq!(objects.len(), 2);

    // Each base entry should have grib keys.
    for (i, entry) in meta.base.iter().enumerate() {
        assert!(entry.contains_key("grib"), "base[{i}] must have grib");
        // geography should be present in each entry (same grid for both).
        let grib = entry.get("grib").unwrap();
        assert!(
            cbor_map_get(grib, "geography").is_some(),
            "base[{i}] grib.geography should be present"
        );
    }

    // parameter namespace should be in each entry (shortName differs).
    for (i, entry) in meta.base.iter().enumerate() {
        let grib = entry.get("grib").unwrap();
        assert!(
            cbor_map_get(grib, "parameter").is_some(),
            "base[{i}] grib.parameter should be present"
        );
    }
}

// ──────────────────────────────────────────────────────────────────────
// 10. test_all_keys_off_no_grib
// ──────────────────────────────────────────────────────────────────────

/// Default options (preserve_all_keys=false) should NOT produce a "grib" key.
#[test]
fn test_all_keys_off_no_grib() {
    let path = testdata().join("2t.grib2");
    let opts = ConvertOptions::default(); // preserve_all_keys = false
    let messages = convert_grib_file(&path, &opts).expect("convert 2t.grib2");

    let (meta, _) = decode(&messages[0], &decode_opts()).expect("decode");

    let entry = &meta.base[0];
    assert!(
        !entry.contains_key("grib"),
        "default options must not produce 'grib' key"
    );
    // mars should still be there
    assert!(entry.contains_key("mars"), "mars must always be present");
}

// ──────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────

/// Concatenate multiple GRIB fixture files into a single temp file.
fn make_combined_grib(filenames: &[&str]) -> tempfile::NamedTempFile {
    use std::io::Write;
    let mut combined = tempfile::NamedTempFile::new().expect("create temp file");
    for name in filenames {
        let data = std::fs::read(testdata().join(name)).expect("read fixture");
        combined.write_all(&data).expect("write to temp file");
    }
    combined.flush().expect("flush");
    combined
}

// ──────────────────────────────────────────────────────────────────────
// 11. test_one_to_one_preserve_all_keys
// ──────────────────────────────────────────────────────────────────────

/// OneToOne with preserve_all_keys: each single-object message has both
/// mars and grib keys in base[0].
#[test]
fn test_one_to_one_preserve_all_keys() {
    let path = testdata().join("2t.grib2");
    let opts = ConvertOptions {
        grouping: Grouping::OneToOne,
        preserve_all_keys: true,
        ..ConvertOptions::default()
    };
    let messages = convert_grib_file(&path, &opts).expect("convert 2t.grib2 one-to-one all-keys");

    assert_eq!(messages.len(), 1, "single GRIB -> single Tensogram message");

    let (meta, objects) = decode(&messages[0], &decode_opts()).expect("decode");
    assert_eq!(objects.len(), 1);
    assert_eq!(meta.base.len(), 1);

    let entry = &meta.base[0];
    assert!(entry.contains_key("mars"), "base[0] must have mars");
    assert!(
        entry.contains_key("grib"),
        "base[0] must have grib with preserve_all_keys"
    );

    // grib should contain geography namespace at minimum
    let grib = entry.get("grib").unwrap();
    assert!(
        cbor_map_get(grib, "geography").is_some(),
        "grib.geography must be present"
    );
}

// ──────────────────────────────────────────────────────────────────────
// 12. test_one_to_one_no_preserve_all_keys
// ──────────────────────────────────────────────────────────────────────

/// OneToOne without preserve_all_keys: base[0] has mars but NOT grib.
#[test]
fn test_one_to_one_no_preserve_all_keys() {
    let path = testdata().join("lsm.grib2");
    let opts = ConvertOptions {
        grouping: Grouping::OneToOne,
        preserve_all_keys: false,
        ..ConvertOptions::default()
    };
    let messages = convert_grib_file(&path, &opts).expect("convert lsm.grib2 one-to-one");

    assert_eq!(messages.len(), 1);

    let (meta, _) = decode(&messages[0], &decode_opts()).expect("decode");
    let entry = &meta.base[0];
    assert!(entry.contains_key("mars"), "mars must be present");
    assert!(
        !entry.contains_key("grib"),
        "grib must NOT be present without preserve_all_keys"
    );
}

// ──────────────────────────────────────────────────────────────────────
// 13. test_merge_all_independent_base_entries
// ──────────────────────────────────────────────────────────────────────

/// Verify each base[i] is self-contained with ALL metadata for that object.
/// With 2 GRIB messages, base[0] and base[1] should each have mars keys
/// independently (not referencing a common section).
#[test]
fn test_merge_all_independent_base_entries() {
    let combined = make_combined_grib(&["lsm.grib2", "2t.grib2"]);
    let opts = ConvertOptions::default();
    let messages = convert_grib_file(combined.path(), &opts).expect("convert merged");

    let (meta, _) = decode(&messages[0], &decode_opts()).expect("decode");
    assert_eq!(meta.base.len(), 2);

    // Both base entries must independently have "mars" with "class" key
    for (i, entry) in meta.base.iter().enumerate() {
        let mars = entry
            .get("mars")
            .unwrap_or_else(|| panic!("base[{i}] must have mars"));
        assert!(
            cbor_map_get(mars, "class").is_some(),
            "base[{i}] mars.class must be independently present"
        );
        assert!(
            cbor_map_get(mars, "grid").is_some(),
            "base[{i}] mars.grid must be independently present"
        );
    }

    // The two entries can have different param values
    let mars0 = meta.base[0].get("mars").unwrap();
    let mars1 = meta.base[1].get("mars").unwrap();
    let param0 = cbor_map_get(mars0, "param").or(cbor_map_get(mars0, "shortName"));
    let param1 = cbor_map_get(mars1, "param").or(cbor_map_get(mars1, "shortName"));
    // lsm and 2t should have different param identifiers
    assert_ne!(
        param0, param1,
        "lsm and 2t should have different parameter identification"
    );
}

// ──────────────────────────────────────────────────────────────────────
// 14. test_one_to_one_multi_grib_messages
// ──────────────────────────────────────────────────────────────────────

/// OneToOne with 4 different GRIB files -> 4 independent Tensogram messages.
#[test]
fn test_one_to_one_multi_grib_messages() {
    let combined = make_combined_grib(&["lsm.grib2", "2t.grib2", "q_150.grib2", "t_600.grib2"]);
    let opts = ConvertOptions {
        grouping: Grouping::OneToOne,
        ..ConvertOptions::default()
    };
    let messages = convert_grib_file(combined.path(), &opts).expect("convert 4 GRIBs");

    assert_eq!(messages.len(), 4, "OneToOne with 4 GRIBs -> 4 messages");

    for (i, msg_bytes) in messages.iter().enumerate() {
        let (meta, objects) = decode(msg_bytes, &decode_opts()).expect("decode");
        assert_eq!(objects.len(), 1, "message {i} has 1 object");
        assert_eq!(meta.base.len(), 1, "message {i} has 1 base entry");

        let entry = &meta.base[0];
        assert!(
            entry.contains_key("mars"),
            "message {i} base[0] must have mars"
        );
        assert!(
            entry.contains_key("_reserved_"),
            "message {i} base[0] must have _reserved_"
        );

        // Verify tensor info in _reserved_
        let reserved = entry.get("_reserved_").unwrap();
        let tensor = cbor_map_get(reserved, "tensor").expect("must have tensor");
        assert!(
            cbor_map_get(tensor, "ndim").is_some(),
            "message {i} must have ndim"
        );
    }
}

// ──────────────────────────────────────────────────────────────────────
// 15. test_merge_all_preserve_all_keys_multi
// ──────────────────────────────────────────────────────────────────────

/// MergeAll + preserve_all_keys with 4 GRIB messages -> 1 Tensogram
/// message with 4 data objects, each base[i] has both mars and grib.
#[test]
fn test_merge_all_preserve_all_keys_multi() {
    let combined = make_combined_grib(&["lsm.grib2", "2t.grib2", "q_150.grib2", "t_600.grib2"]);
    let opts = ConvertOptions {
        grouping: Grouping::MergeAll,
        preserve_all_keys: true,
        ..ConvertOptions::default()
    };
    let messages = convert_grib_file(combined.path(), &opts).expect("convert 4 GRIBs merged");

    assert_eq!(messages.len(), 1);

    let (meta, objects) = decode(&messages[0], &decode_opts()).expect("decode");
    assert_eq!(objects.len(), 4, "4 GRIB -> 4 data objects");
    assert_eq!(meta.base.len(), 4, "4 base entries");

    for (i, entry) in meta.base.iter().enumerate() {
        assert!(entry.contains_key("mars"), "base[{i}] must have mars");
        assert!(entry.contains_key("grib"), "base[{i}] must have grib");
        assert!(
            entry.contains_key("_reserved_"),
            "base[{i}] must have _reserved_"
        );
    }
}

// ──────────────────────────────────────────────────────────────────────
// 16. test_build_data_object_strides
// ──────────────────────────────────────────────────────────────────────

/// Verify that build_data_object computes correct row-major strides for
/// the 2D GRIB grids.
#[test]
fn test_2d_strides_correct() {
    let path = testdata().join("2t.grib2");
    let opts = ConvertOptions::default();
    let messages = convert_grib_file(&path, &opts).expect("convert");

    let (_, objects) = decode(&messages[0], &decode_opts()).expect("decode");
    let (desc, _) = &objects[0];

    // 2D grid: [nj=721, ni=1440], strides should be [1440, 1]
    assert_eq!(desc.shape, vec![721, 1440]);
    assert_eq!(desc.strides, vec![1440, 1]);
}

// ──────────────────────────────────────────────────────────────────────
// 17. test_empty_grib_file_error
// ──────────────────────────────────────────────────────────────────────

/// An empty file (no GRIB messages) should return GribError::NoMessages.
#[test]
fn test_empty_grib_file_error() {
    use std::io::Write;
    let mut empty = tempfile::NamedTempFile::new().expect("create temp file");
    // Write some garbage that isn't a GRIB message
    empty.write_all(b"not a grib file").expect("write");
    empty.flush().expect("flush");

    let opts = ConvertOptions::default();
    let result = convert_grib_file(empty.path(), &opts);
    // ecCodes should find no messages or return an error
    assert!(result.is_err(), "empty/invalid GRIB file should fail");
}
