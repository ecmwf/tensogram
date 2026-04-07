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

pub(crate) fn attr_value_to_cbor(val: &AttributeValue) -> Option<CborValue> {
    match val {
        AttributeValue::Str(s) => Some(CborValue::Text(s.clone())),
        AttributeValue::Strs(v) => Some(CborValue::Array(
            v.iter().map(|s| CborValue::Text(s.clone())).collect(),
        )),
        AttributeValue::Double(f) => Some(CborValue::Float(*f)),
        AttributeValue::Doubles(v) => Some(CborValue::Array(
            v.iter().map(|f| CborValue::Float(*f)).collect(),
        )),
        AttributeValue::Float(f) => Some(CborValue::Float(*f as f64)),
        AttributeValue::Floats(v) => Some(CborValue::Array(
            v.iter().map(|f| CborValue::Float(*f as f64)).collect(),
        )),
        AttributeValue::Int(i) => Some(CborValue::Integer((*i as i64).into())),
        AttributeValue::Ints(v) => Some(CborValue::Array(
            v.iter()
                .map(|i| CborValue::Integer((*i as i64).into()))
                .collect(),
        )),
        AttributeValue::Short(i) => Some(CborValue::Integer((*i as i64).into())),
        AttributeValue::Shorts(v) => Some(CborValue::Array(
            v.iter()
                .map(|i| CborValue::Integer((*i as i64).into()))
                .collect(),
        )),
        AttributeValue::Ushort(i) => Some(CborValue::Integer((*i as i64).into())),
        AttributeValue::Ushorts(v) => Some(CborValue::Array(
            v.iter()
                .map(|i| CborValue::Integer((*i as i64).into()))
                .collect(),
        )),
        AttributeValue::Uint(i) => Some(CborValue::Integer((*i as i64).into())),
        AttributeValue::Uints(v) => Some(CborValue::Array(
            v.iter()
                .map(|i| CborValue::Integer((*i as i64).into()))
                .collect(),
        )),
        AttributeValue::Longlong(i) => Some(CborValue::Integer((*i).into())),
        AttributeValue::Longlongs(v) => Some(CborValue::Array(
            v.iter().map(|i| CborValue::Integer((*i).into())).collect(),
        )),
        AttributeValue::Ulonglong(i) => Some(CborValue::Integer((*i as i64).into())),
        AttributeValue::Ulonglongs(v) => Some(CborValue::Array(
            v.iter()
                .map(|i| CborValue::Integer((*i as i64).into()))
                .collect(),
        )),
        AttributeValue::Schar(i) => Some(CborValue::Integer((*i as i64).into())),
        AttributeValue::Schars(v) => Some(CborValue::Array(
            v.iter()
                .map(|i| CborValue::Integer((*i as i64).into()))
                .collect(),
        )),
        AttributeValue::Uchar(i) => Some(CborValue::Integer((*i as i64).into())),
        AttributeValue::Uchars(v) => Some(CborValue::Array(
            v.iter()
                .map(|i| CborValue::Integer((*i as i64).into()))
                .collect(),
        )),
    }
}

pub(crate) fn extract_var_attrs(var: &netcdf::Variable<'_>) -> BTreeMap<String, CborValue> {
    let mut map = BTreeMap::new();
    for attr in var.attributes() {
        if let Ok(val) = attr.value() {
            if let Some(cbor) = attr_value_to_cbor(&val) {
                map.insert(attr.name().to_string(), cbor);
            }
        }
    }
    map
}

pub(crate) fn extract_cf_attrs(var: &netcdf::Variable<'_>) -> BTreeMap<String, CborValue> {
    let mut map = BTreeMap::new();
    for attr in var.attributes() {
        let name = attr.name();
        if CF_ATTRIBUTES.contains(&name) {
            if let Ok(val) = attr.value() {
                if let Some(cbor) = attr_value_to_cbor(&val) {
                    map.insert(name.to_string(), cbor);
                }
            }
        }
    }
    map
}
