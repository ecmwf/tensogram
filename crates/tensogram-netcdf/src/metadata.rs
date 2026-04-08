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
        AttributeValue::Ulonglong(i) => CborValue::Integer((*i as i64).into()),
        AttributeValue::Ulonglongs(v) => CborValue::Array(
            v.iter()
                .map(|i| CborValue::Integer((*i as i64).into()))
                .collect(),
        ),
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
