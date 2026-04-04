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

/// Extract the `common["mars"]` sub-map from global metadata.
fn get_common_mars(meta: &tensogram_core::GlobalMetadata) -> &CborValue {
    meta.common
        .get("mars")
        .expect("common must contain 'mars' key")
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

    // MARS keys should be under common["mars"].
    let mars = get_common_mars(&meta);
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
    let mars = get_common_mars(&meta);
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
/// message with 2 objects. Common mars keys in common["mars"], varying
/// in payload[i]["mars"].
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

    // Common mars keys should be shared (class, type, stream, date, etc.).
    let mars = get_common_mars(&meta);
    assert!(
        cbor_map_get(mars, "class").is_some(),
        "'class' should be in common mars"
    );

    // Per-object payload entries should have varying mars keys.
    assert_eq!(meta.payload.len(), 2, "payload should have 2 entries");

    // At least one entry should have per-object mars keys (param differs).
    let has_varying_mars = meta.payload.iter().any(|entry| entry.contains_key("mars"));
    assert!(
        has_varying_mars,
        "payload entries should carry varying mars keys"
    );

    // Both entries should have auto-populated ndim/shape/strides/dtype.
    for (i, entry) in meta.payload.iter().enumerate() {
        assert!(entry.contains_key("ndim"), "payload[{}] must have ndim", i);
        assert!(
            entry.contains_key("dtype"),
            "payload[{}] must have dtype",
            i
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
        // Each message should have all MARS keys in common["mars"].
        assert!(
            meta.common.contains_key("mars"),
            "message {} common must have mars",
            i
        );
    }
}

// ──────────────────────────────────────────────────────────────────────
// 7. test_payload_objects_metadata
// ──────────────────────────────────────────────────────────────────────

/// Verify payload entries have ndim, shape, strides, dtype, and mars.grid.
#[test]
fn test_payload_objects_metadata() {
    let path = testdata().join("2t.grib2");
    let opts = ConvertOptions::default();
    let messages = convert_grib_file(&path, &opts).expect("convert 2t.grib2");

    let (meta, _) = decode(&messages[0], &decode_opts()).expect("decode");

    // payload should have one entry.
    assert_eq!(meta.payload.len(), 1, "single object -> one payload entry");

    let entry = &meta.payload[0];

    // Required auto-populated fields: ndim, shape, strides, dtype.
    assert!(entry.contains_key("ndim"), "must have ndim");
    assert!(entry.contains_key("shape"), "must have shape");
    assert!(entry.contains_key("strides"), "must have strides");
    assert_eq!(
        entry.get("dtype"),
        Some(&CborValue::Text("float64".to_string())),
        "dtype should be 'float64'"
    );

    // For a single-object message, all mars keys (including grid) are in
    // common["mars"], not in payload[0]["mars"]. Verify grid is in common.
    let mars = get_common_mars(&meta);
    assert_eq!(
        cbor_map_get(mars, "grid"),
        Some(&CborValue::Text("regular_ll".to_string())),
        "common mars.grid should be 'regular_ll'"
    );
}

// ──────────────────────────────────────────────────────────────────────
// 8. test_all_keys_single_object
// ──────────────────────────────────────────────────────────────────────

/// With `preserve_all_keys`, common["grib"] should contain namespace sub-maps.
#[test]
fn test_all_keys_single_object() {
    let path = testdata().join("2t.grib2");
    let opts = ConvertOptions {
        preserve_all_keys: true,
        ..ConvertOptions::default()
    };
    let messages = convert_grib_file(&path, &opts).expect("convert 2t.grib2 all-keys");

    let (meta, _) = decode(&messages[0], &decode_opts()).expect("decode");

    // mars keys still in common["mars"]
    assert!(meta.common.contains_key("mars"), "must have mars");

    // grib namespace keys in common["grib"]
    let grib = meta
        .common
        .get("grib")
        .expect("common must contain 'grib' when preserve_all_keys is on");

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

/// With `preserve_all_keys` and MergeAll, grib namespace keys partition
/// into common["grib"] (shared) and payload[i]["grib"] (varying).
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

    // Common grib keys should be present (e.g. geography is identical).
    assert!(
        meta.common.contains_key("grib"),
        "merged common must have grib"
    );
    let grib = &meta.common["grib"];
    assert!(
        cbor_map_get(grib, "geography").is_some(),
        "geography should be common (same grid)"
    );

    // Varying grib keys should be in payload entries.
    // parameter.shortName differs (lsm vs 2t), so parameter should appear
    // in at least one payload entry's grib map.
    let has_varying_grib = meta.payload.iter().any(|e| e.contains_key("grib"));
    assert!(
        has_varying_grib,
        "payload entries should have varying grib keys"
    );

    // Statistics always varies per object.
    let has_statistics_in_payload = meta.payload.iter().any(|e| {
        e.get("grib")
            .and_then(|g| cbor_map_get(g, "statistics"))
            .is_some()
    });
    assert!(
        has_statistics_in_payload,
        "statistics should be per-object (always varies)"
    );
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

    assert!(
        !meta.common.contains_key("grib"),
        "default options must not produce 'grib' key"
    );
    // mars should still be there
    assert!(
        meta.common.contains_key("mars"),
        "mars must always be present"
    );
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
