// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::collections::BTreeMap;

use ciborium::Value as CborValue;
use netcdf::AttributeValue;

pub(crate) const CF_ATTRIBUTES: &[&str] = &[
    "standard_name",
    "long_name",
    "units",
    "calendar",
    "cell_methods",
    "coordinates",
    "axis",
    "positive",
    "valid_min",
    "valid_max",
    "valid_range",
    "bounds",
    "grid_mapping",
    "ancillary_variables",
    "flag_values",
    "flag_meanings",
];

/// Map a NetCDF attribute value to its CBOR equivalent.
///
/// The match is exhaustive over every `AttributeValue` variant the
/// `netcdf` crate currently exposes, so adding a new variant upstream
/// will produce a compile error here rather than silently dropping
/// data. Numeric scalars are widened to `i64` / `f64` to fit CBOR's
/// integer and float types.
pub(crate) fn attr_value_to_cbor(val: &AttributeValue) -> CborValue {
    match val {
        AttributeValue::Str(s) => CborValue::Text(s.clone()),
        AttributeValue::Strs(v) => {
            CborValue::Array(v.iter().map(|s| CborValue::Text(s.clone())).collect())
        }
        AttributeValue::Double(f) => CborValue::Float(*f),
        AttributeValue::Doubles(v) => {
            CborValue::Array(v.iter().map(|f| CborValue::Float(*f)).collect())
        }
        AttributeValue::Float(f) => CborValue::Float(*f as f64),
        AttributeValue::Floats(v) => {
            CborValue::Array(v.iter().map(|f| CborValue::Float(*f as f64)).collect())
        }
        AttributeValue::Int(i) => CborValue::Integer((*i as i64).into()),
        AttributeValue::Ints(v) => CborValue::Array(
            v.iter()
                .map(|i| CborValue::Integer((*i as i64).into()))
                .collect(),
        ),
        AttributeValue::Short(i) => CborValue::Integer((*i as i64).into()),
        AttributeValue::Shorts(v) => CborValue::Array(
            v.iter()
                .map(|i| CborValue::Integer((*i as i64).into()))
                .collect(),
        ),
        AttributeValue::Ushort(i) => CborValue::Integer((*i as i64).into()),
        AttributeValue::Ushorts(v) => CborValue::Array(
            v.iter()
                .map(|i| CborValue::Integer((*i as i64).into()))
                .collect(),
        ),
        AttributeValue::Uint(i) => CborValue::Integer((*i as i64).into()),
        AttributeValue::Uints(v) => CborValue::Array(
            v.iter()
                .map(|i| CborValue::Integer((*i as i64).into()))
                .collect(),
        ),
        AttributeValue::Longlong(i) => CborValue::Integer((*i).into()),
        AttributeValue::Longlongs(v) => {
            CborValue::Array(v.iter().map(|i| CborValue::Integer((*i).into())).collect())
        }
        AttributeValue::Ulonglong(i) => CborValue::Integer((*i).into()),
        AttributeValue::Ulonglongs(v) => {
            CborValue::Array(v.iter().map(|i| CborValue::Integer((*i).into())).collect())
        }
        AttributeValue::Schar(i) => CborValue::Integer((*i as i64).into()),
        AttributeValue::Schars(v) => CborValue::Array(
            v.iter()
                .map(|i| CborValue::Integer((*i as i64).into()))
                .collect(),
        ),
        AttributeValue::Uchar(i) => CborValue::Integer((*i as i64).into()),
        AttributeValue::Uchars(v) => CborValue::Array(
            v.iter()
                .map(|i| CborValue::Integer((*i as i64).into()))
                .collect(),
        ),
    }
}

/// Extract every variable attribute as a CBOR map.
///
/// If an attribute's value cannot be read (very rare — only happens
/// with corrupt files or unsupported upstream types), a warning is
/// emitted to stderr so the user can see the drop. The function
/// itself is infallible by design.
pub(crate) fn extract_var_attrs(var: &netcdf::Variable<'_>) -> BTreeMap<String, CborValue> {
    let mut map = BTreeMap::new();
    let var_name = var.name();
    for attr in var.attributes() {
        let name = attr.name();
        match attr.value() {
            Ok(val) => {
                map.insert(name.to_string(), attr_value_to_cbor(&val));
            }
            Err(e) => {
                eprintln!("warning: variable '{var_name}': failed to read attribute '{name}': {e}");
            }
        }
    }
    map
}

/// Extract the CF allow-list attributes as a CBOR map. Unlike
/// [`extract_var_attrs`] this only retains the 16 entries listed in
/// [`CF_ATTRIBUTES`]; anything else is ignored (not warned about,
/// because it's expected that variables carry many non-CF attributes).
pub(crate) fn extract_cf_attrs(var: &netcdf::Variable<'_>) -> BTreeMap<String, CborValue> {
    let mut map = BTreeMap::new();
    let var_name = var.name();
    for attr in var.attributes() {
        let name = attr.name();
        if !CF_ATTRIBUTES.contains(&name) {
            continue;
        }
        match attr.value() {
            Ok(val) => {
                map.insert(name.to_string(), attr_value_to_cbor(&val));
            }
            Err(e) => {
                eprintln!(
                    "warning: variable '{var_name}': failed to read CF attribute '{name}': {e}"
                );
            }
        }
    }
    map
}

// ── Unit tests ──────────────────────────────────────────────────────────────
//
// Exhaustively cover every `AttributeValue` → `CborValue` mapping so the
// match in `attr_value_to_cbor` never silently diverges from the netcdf
// crate's variants. These tests don't need fixture files because
// `AttributeValue` is a public enum with ordinary constructors.

#[cfg(test)]
mod tests {
    use super::*;

    fn expect_text(val: &CborValue, want: &str) {
        match val {
            CborValue::Text(s) => assert_eq!(s, want),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    fn expect_float(val: &CborValue, want: f64) {
        match val {
            CborValue::Float(f) => assert!((*f - want).abs() < 1e-12, "expected {want}, got {f}"),
            other => panic!("expected Float, got {other:?}"),
        }
    }

    fn expect_int(val: &CborValue, want: i64) {
        match val {
            CborValue::Integer(i) => {
                let n: i128 = (*i).into();
                assert_eq!(n, want as i128);
            }
            other => panic!("expected Integer, got {other:?}"),
        }
    }

    fn expect_array_len(val: &CborValue, want: usize) -> &Vec<CborValue> {
        match val {
            CborValue::Array(arr) => {
                assert_eq!(arr.len(), want);
                arr
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }

    // ── String variants ────────────────────────────────────────────────
    #[test]
    fn attr_value_to_cbor_str() {
        expect_text(
            &attr_value_to_cbor(&AttributeValue::Str("K".to_string())),
            "K",
        );
    }

    #[test]
    fn attr_value_to_cbor_strs() {
        let v = attr_value_to_cbor(&AttributeValue::Strs(vec![
            "a".to_string(),
            "bb".to_string(),
        ]));
        let arr = expect_array_len(&v, 2);
        expect_text(&arr[0], "a");
        expect_text(&arr[1], "bb");
    }

    // ── Double (f64) ───────────────────────────────────────────────────
    #[test]
    fn attr_value_to_cbor_double() {
        expect_float(&attr_value_to_cbor(&AttributeValue::Double(273.15)), 273.15);
    }

    #[test]
    fn attr_value_to_cbor_doubles() {
        let v = attr_value_to_cbor(&AttributeValue::Doubles(vec![1.0, -2.5, 3.75]));
        let arr = expect_array_len(&v, 3);
        expect_float(&arr[0], 1.0);
        expect_float(&arr[1], -2.5);
        expect_float(&arr[2], 3.75);
    }

    // ── Float (f32 → f64) ──────────────────────────────────────────────
    #[test]
    fn attr_value_to_cbor_float() {
        expect_float(&attr_value_to_cbor(&AttributeValue::Float(2.5_f32)), 2.5);
    }

    #[test]
    fn attr_value_to_cbor_floats() {
        let v = attr_value_to_cbor(&AttributeValue::Floats(vec![1.0_f32, 2.0_f32]));
        let arr = expect_array_len(&v, 2);
        expect_float(&arr[0], 1.0);
        expect_float(&arr[1], 2.0);
    }

    // ── Int (i32 → i64) ────────────────────────────────────────────────
    #[test]
    fn attr_value_to_cbor_int() {
        expect_int(&attr_value_to_cbor(&AttributeValue::Int(-42_i32)), -42);
    }

    #[test]
    fn attr_value_to_cbor_ints() {
        let v = attr_value_to_cbor(&AttributeValue::Ints(vec![1_i32, -1_i32, 999_i32]));
        let arr = expect_array_len(&v, 3);
        expect_int(&arr[0], 1);
        expect_int(&arr[1], -1);
        expect_int(&arr[2], 999);
    }

    // ── Short (i16 → i64) ──────────────────────────────────────────────
    #[test]
    fn attr_value_to_cbor_short() {
        expect_int(
            &attr_value_to_cbor(&AttributeValue::Short(-32768_i16)),
            -32768,
        );
    }

    #[test]
    fn attr_value_to_cbor_shorts() {
        let v = attr_value_to_cbor(&AttributeValue::Shorts(vec![0_i16, -1_i16]));
        let arr = expect_array_len(&v, 2);
        expect_int(&arr[0], 0);
        expect_int(&arr[1], -1);
    }

    // ── Ushort (u16 → i64) ─────────────────────────────────────────────
    #[test]
    fn attr_value_to_cbor_ushort() {
        expect_int(
            &attr_value_to_cbor(&AttributeValue::Ushort(65535_u16)),
            65535,
        );
    }

    #[test]
    fn attr_value_to_cbor_ushorts() {
        let v = attr_value_to_cbor(&AttributeValue::Ushorts(vec![1_u16, 65535_u16]));
        let arr = expect_array_len(&v, 2);
        expect_int(&arr[0], 1);
        expect_int(&arr[1], 65535);
    }

    // ── Uint (u32 → i64) ───────────────────────────────────────────────
    #[test]
    fn attr_value_to_cbor_uint() {
        expect_int(
            &attr_value_to_cbor(&AttributeValue::Uint(4_000_000_000_u32)),
            4_000_000_000,
        );
    }

    #[test]
    fn attr_value_to_cbor_uints() {
        let v = attr_value_to_cbor(&AttributeValue::Uints(vec![0_u32, 1_000_000_u32]));
        let arr = expect_array_len(&v, 2);
        expect_int(&arr[0], 0);
        expect_int(&arr[1], 1_000_000);
    }

    // ── Longlong (i64) ─────────────────────────────────────────────────
    #[test]
    fn attr_value_to_cbor_longlong() {
        expect_int(
            &attr_value_to_cbor(&AttributeValue::Longlong(i64::MIN)),
            i64::MIN,
        );
    }

    #[test]
    fn attr_value_to_cbor_longlongs() {
        let v = attr_value_to_cbor(&AttributeValue::Longlongs(vec![0_i64, i64::MAX, i64::MIN]));
        let arr = expect_array_len(&v, 3);
        expect_int(&arr[0], 0);
        expect_int(&arr[1], i64::MAX);
        expect_int(&arr[2], i64::MIN);
    }

    // ── Ulonglong (u64 — full range, no i64 wrap-around) ──────────────
    #[test]
    fn attr_value_to_cbor_ulonglong() {
        // Small u64 that fits in i64 round-trips cleanly.
        expect_int(
            &attr_value_to_cbor(&AttributeValue::Ulonglong(12345_u64)),
            12345,
        );
    }

    #[test]
    fn attr_value_to_cbor_ulonglong_above_i64_max() {
        // Regression test: u64 values above i64::MAX must NOT wrap
        // around into negatives. Ciborium's Integer has a native
        // From<u64> impl that represents the full u64 range, so the
        // conversion is now `(*i).into()` not `(*i as i64).into()`.
        let large = (i64::MAX as u64) + 42;
        let cbor = attr_value_to_cbor(&AttributeValue::Ulonglong(large));
        match cbor {
            CborValue::Integer(i) => {
                let n: i128 = i.into();
                assert_eq!(n, large as i128);
                assert!(n > 0, "u64 > i64::MAX must not wrap to negative");
            }
            other => panic!("expected Integer, got {other:?}"),
        }
    }

    #[test]
    fn attr_value_to_cbor_ulonglongs() {
        let v = attr_value_to_cbor(&AttributeValue::Ulonglongs(vec![
            0_u64,
            42_u64,
            1_000_000_u64,
            // Include a value above i64::MAX to pin the no-wrap
            // behaviour at the array variant too.
            u64::MAX,
        ]));
        let arr = expect_array_len(&v, 4);
        expect_int(&arr[0], 0);
        expect_int(&arr[1], 42);
        expect_int(&arr[2], 1_000_000);
        match &arr[3] {
            CborValue::Integer(i) => {
                let n: i128 = (*i).into();
                assert_eq!(n, u64::MAX as i128);
            }
            other => panic!("expected Integer for u64::MAX, got {other:?}"),
        }
    }

    // ── Schar (i8 → i64) ───────────────────────────────────────────────
    #[test]
    fn attr_value_to_cbor_schar() {
        expect_int(&attr_value_to_cbor(&AttributeValue::Schar(-128_i8)), -128);
    }

    #[test]
    fn attr_value_to_cbor_schars() {
        let v = attr_value_to_cbor(&AttributeValue::Schars(vec![-1_i8, 0_i8, 127_i8]));
        let arr = expect_array_len(&v, 3);
        expect_int(&arr[0], -1);
        expect_int(&arr[1], 0);
        expect_int(&arr[2], 127);
    }

    // ── Uchar (u8 → i64) ───────────────────────────────────────────────
    #[test]
    fn attr_value_to_cbor_uchar() {
        expect_int(&attr_value_to_cbor(&AttributeValue::Uchar(255_u8)), 255);
    }

    #[test]
    fn attr_value_to_cbor_uchars() {
        let v = attr_value_to_cbor(&AttributeValue::Uchars(vec![0_u8, 128_u8, 255_u8]));
        let arr = expect_array_len(&v, 3);
        expect_int(&arr[0], 0);
        expect_int(&arr[1], 128);
        expect_int(&arr[2], 255);
    }

    // ── CF allow-list membership ───────────────────────────────────────

    #[test]
    fn cf_attributes_contains_all_16() {
        // Guard against accidental deletions / renames of the allow-list.
        assert_eq!(CF_ATTRIBUTES.len(), 16);
        for expected in &[
            "standard_name",
            "long_name",
            "units",
            "calendar",
            "cell_methods",
            "coordinates",
            "axis",
            "positive",
            "valid_min",
            "valid_max",
            "valid_range",
            "bounds",
            "grid_mapping",
            "ancillary_variables",
            "flag_values",
            "flag_meanings",
        ] {
            assert!(
                CF_ATTRIBUTES.contains(expected),
                "CF_ATTRIBUTES should contain '{expected}'"
            );
        }
    }
}
