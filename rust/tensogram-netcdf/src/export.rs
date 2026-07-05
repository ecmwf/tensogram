// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! NetCDF export: reconstruct a NetCDF file from a Tensogram message.
//!
//! Reverse of [`crate::convert_netcdf_file`], using only the **safe** `netcdf`
//! crate (no `netcdf-sys` FFI).  Reads the file-level dimension registry
//! (`base[i]["netcdf"]["_file"]`), per-variable dim names (`_dims`), native
//! dtype (from the descriptor), and attributes, and rebuilds the file.
//! See `plans/GRIB_NETCDF_ROUNDTRIP.md`.
//!
//! Scope: dimensions + variables (native dtype) + data + attributes (scalar and
//! array, with their exact NetCDF types restored from the `_attr_types` /
//! `_global_types` sidecar), emitted as netCDF-4.  NaN / missing values ride
//! back in via `decode`'s non-finite restoration.  Classic-format byte-matching,
//! groups, chunking/compression, and CF re-packing are follow-ups.

use std::path::Path;

use ciborium::Value as CborValue;
use netcdf::AttributeValue;

use tensogram::types::{ByteOrder, GlobalMetadata};
use tensogram::{decode, DecodeOptions, Dtype};

use crate::error::NetcdfError;

/// Attribute keys used internally by the importer to carry structure, not real
/// NetCDF attributes — they must not be written back as attributes.
///
/// This list must mirror the structural keys inserted by the importer
/// (`convert::extract_variable` / `extract_variable_record`).  Drift is caught
/// by the round-trip tests: a leaked internal key surfaces as a bogus attribute
/// (`roundtrip_attr_types_exact`, the `ncdump` e2e), and a mis-filtered real
/// attribute goes missing.
const INTERNAL_KEYS: &[&str] = &[
    "_dims",
    "_file",
    "_global",
    "_attr_types",
    "_global_types",
    "record_index",
];

/// Reconstruct a NetCDF file at `out_path` from a Tensogram message produced by
/// [`crate::convert_netcdf_file`].
///
/// # Errors
///
/// - [`NetcdfError::InvalidData`] — the message is not decodable, or lacks the
///   structural metadata (`name` / `_dims` / `_file`) written by `convert-netcdf`.
/// - [`NetcdfError::Netcdf`] — libnetcdf rejected a definition or write.
pub fn to_netcdf(message: &[u8], out_path: &Path) -> Result<(), NetcdfError> {
    let (meta, objects) = decode(message, &DecodeOptions::default())
        .map_err(|e| NetcdfError::InvalidData(format!("decode tensogram message: {e}")))?;

    let mut file = netcdf::create(out_path)?;

    // Dimensions from the file registry (identical across every object).
    for (name, len, unlimited) in file_dims(&meta)? {
        if unlimited {
            file.add_unlimited_dimension(&name)?;
        } else {
            file.add_dimension(&name, len)?;
        }
    }

    // Global attributes (exact types via the `_global_types` sidecar).
    for (name, av) in global_attrs(&meta) {
        file.add_attribute(&name, av)?;
    }

    for (i, (desc, payload)) in objects.iter().enumerate() {
        let name = var_name(&meta, i)?;
        let dim_names = var_dim_names(&meta, i)?;
        let dim_refs: Vec<&str> = dim_names.iter().map(String::as_str).collect();
        let attrs = var_attrs(&meta, i);
        add_variable(
            &mut file,
            &name,
            &dim_refs,
            desc.dtype,
            desc.byte_order,
            payload,
            &attrs,
        )?;
    }

    Ok(())
}

/// Add one variable (dtype-dispatched), write its data, and set its attributes.
fn add_variable(
    file: &mut netcdf::FileMut,
    name: &str,
    dims: &[&str],
    dtype: Dtype,
    order: ByteOrder,
    payload: &[u8],
    attrs: &[(String, AttributeValue)],
) -> Result<(), NetcdfError> {
    macro_rules! arm {
        ($t:ty) => {{
            const SZ: usize = std::mem::size_of::<$t>();
            let mut vals: Vec<$t> = Vec::with_capacity(payload.len() / SZ);
            for c in payload.chunks_exact(SZ) {
                let mut b = [0u8; SZ];
                b.copy_from_slice(c);
                vals.push(match order {
                    ByteOrder::Little => <$t>::from_le_bytes(b),
                    ByteOrder::Big => <$t>::from_be_bytes(b),
                });
            }
            let mut var = file.add_variable::<$t>(name, dims)?;
            for (an, av) in attrs {
                var.put_attribute(an, av.clone())?;
            }
            var.put_values(&vals, ..)?;
        }};
    }

    match dtype {
        Dtype::Int8 => arm!(i8),
        Dtype::Uint8 => arm!(u8),
        Dtype::Int16 => arm!(i16),
        Dtype::Uint16 => arm!(u16),
        Dtype::Int32 => arm!(i32),
        Dtype::Uint32 => arm!(u32),
        Dtype::Int64 => arm!(i64),
        Dtype::Uint64 => arm!(u64),
        Dtype::Float32 => arm!(f32),
        Dtype::Float64 => arm!(f64),
        other => {
            return Err(NetcdfError::InvalidData(format!(
                "to-netcdf: unsupported dtype {other} for variable '{name}'"
            )));
        }
    }
    Ok(())
}

// ── Metadata accessors ───────────────────────────────────────────────────────

/// Look up a string-keyed entry in a CBOR map.
fn map_get<'a>(map: &'a [(CborValue, CborValue)], key: &str) -> Option<&'a CborValue> {
    map.iter()
        .find(|(k, _)| matches!(k, CborValue::Text(s) if s == key))
        .map(|(_, v)| v)
}

/// The `netcdf` sub-map for object `i`.
fn netcdf_map(meta: &GlobalMetadata, i: usize) -> Option<&Vec<(CborValue, CborValue)>> {
    let entry = meta.base.get(i)?;
    match entry.get("netcdf") {
        Some(CborValue::Map(m)) => Some(m),
        _ => None,
    }
}

/// Dimension registry: `[(name, len, unlimited)]` from `base[0].netcdf._file.dims`.
fn file_dims(meta: &GlobalMetadata) -> Result<Vec<(String, usize, bool)>, NetcdfError> {
    let nc = netcdf_map(meta, 0)
        .ok_or_else(|| NetcdfError::InvalidData("no netcdf metadata on object 0".into()))?;
    let Some(CborValue::Map(fm)) = map_get(nc, "_file") else {
        return Err(NetcdfError::InvalidData(
            "missing '_file' dimension registry; was this produced by convert-netcdf?".into(),
        ));
    };
    let Some(CborValue::Array(dims)) = map_get(fm, "dims") else {
        return Err(NetcdfError::InvalidData(
            "'_file' has no 'dims' array".into(),
        ));
    };
    let mut out = Vec::with_capacity(dims.len());
    for d in dims {
        let CborValue::Map(dm) = d else { continue };
        let name = match map_get(dm, "name") {
            Some(CborValue::Text(s)) => s.clone(),
            _ => continue,
        };
        let len = match map_get(dm, "len") {
            Some(CborValue::Integer(i)) => usize::try_from(i128::from(*i)).unwrap_or(0),
            _ => 0,
        };
        let unlimited = matches!(map_get(dm, "unlimited"), Some(CborValue::Bool(true)));
        out.push((name, len, unlimited));
    }
    Ok(out)
}

/// Variable name for object `i` (`base[i]["name"]`).
fn var_name(meta: &GlobalMetadata, i: usize) -> Result<String, NetcdfError> {
    match meta.base.get(i).and_then(|e| e.get("name")) {
        Some(CborValue::Text(s)) => Ok(s.clone()),
        _ => Err(NetcdfError::InvalidData(format!(
            "object {i} has no variable 'name'"
        ))),
    }
}

/// Ordered dimension names for object `i` (`base[i].netcdf._dims`).
fn var_dim_names(meta: &GlobalMetadata, i: usize) -> Result<Vec<String>, NetcdfError> {
    let nc = netcdf_map(meta, i)
        .ok_or_else(|| NetcdfError::InvalidData(format!("object {i} has no netcdf metadata")))?;
    let Some(CborValue::Array(dims)) = map_get(nc, "_dims") else {
        return Err(NetcdfError::InvalidData(format!(
            "object {i} has no '_dims' list"
        )));
    };
    Ok(dims
        .iter()
        .filter_map(|d| match d {
            CborValue::Text(s) => Some(s.clone()),
            _ => None,
        })
        .collect())
}

/// The `name → type-tag` map stored under `key` (`_attr_types` / `_global_types`)
/// in a `netcdf`-style CBOR map, if present.
fn type_tags<'a>(
    map: &'a [(CborValue, CborValue)],
    key: &str,
) -> Option<&'a Vec<(CborValue, CborValue)>> {
    match map_get(map, key) {
        Some(CborValue::Map(m)) => Some(m),
        _ => None,
    }
}

/// The exact-type tag recorded for attribute `name`, if any.
fn tag_for<'a>(tags: Option<&'a Vec<(CborValue, CborValue)>>, name: &str) -> Option<&'a str> {
    let tags = tags?;
    match map_get(tags, name) {
        Some(CborValue::Text(s)) => Some(s.as_str()),
        _ => None,
    }
}

/// Real variable attributes for object `i` (excludes internal `_*` keys),
/// reconstructed with their exact on-disk types via the `_attr_types` sidecar.
fn var_attrs(meta: &GlobalMetadata, i: usize) -> Vec<(String, AttributeValue)> {
    let Some(nc) = netcdf_map(meta, i) else {
        return Vec::new();
    };
    let tags = type_tags(nc, "_attr_types");
    nc.iter()
        .filter_map(|(k, v)| match k {
            CborValue::Text(name) if !INTERNAL_KEYS.contains(&name.as_str()) => {
                cbor_to_attr_typed(v, tag_for(tags, name)).map(|av| (name.clone(), av))
            }
            _ => None,
        })
        .collect()
}

/// Global attributes, read from `base[0].netcdf._global`, reconstructed with
/// their exact types via the `_global_types` sidecar.
fn global_attrs(meta: &GlobalMetadata) -> Vec<(String, AttributeValue)> {
    let Some(nc) = netcdf_map(meta, 0) else {
        return Vec::new();
    };
    let tags = type_tags(nc, "_global_types");
    match map_get(nc, "_global") {
        Some(CborValue::Map(g)) => g
            .iter()
            .filter_map(|(k, v)| match k {
                CborValue::Text(name) => {
                    cbor_to_attr_typed(v, tag_for(tags, name)).map(|av| (name.clone(), av))
                }
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Coerce a CBOR scalar to `f64` (accepts float or integer).
fn as_f64(v: &CborValue) -> Option<f64> {
    match v {
        CborValue::Float(f) => Some(*f),
        CborValue::Integer(i) => Some(i128::from(*i) as f64),
        _ => None,
    }
}

/// Coerce a CBOR scalar to `i64`.
fn as_i64(v: &CborValue) -> Option<i64> {
    match v {
        CborValue::Integer(i) => i64::try_from(*i).ok(),
        CborValue::Float(f) => Some(*f as i64),
        _ => None,
    }
}

/// Coerce a CBOR scalar to `u64` (preserves the full unsigned range).
fn as_u64(v: &CborValue) -> Option<u64> {
    match v {
        CborValue::Integer(i) => u64::try_from(i128::from(*i)).ok(),
        CborValue::Float(f) => Some(*f as u64),
        _ => None,
    }
}

/// CBOR → NetCDF attribute value, restoring the *exact* on-disk type from the
/// `_attr_types` / `_global_types` sidecar tag (see
/// [`crate::metadata::attr_value_type_tag`]).  Scalars and arrays share a tag;
/// array-ness is recovered from the CBOR value shape.  When no tag is present
/// (e.g. metadata predating type capture, or a CF-lifted attribute), falls back
/// to widening — integers to `int64`, floats to `double`.
fn cbor_to_attr_typed(v: &CborValue, tag: Option<&str>) -> Option<AttributeValue> {
    if let CborValue::Array(a) = v {
        return cbor_array_to_attr(a, tag);
    }
    match tag {
        Some("string") => match v {
            CborValue::Text(s) => Some(AttributeValue::Str(s.clone())),
            _ => None,
        },
        Some("double") => as_f64(v).map(AttributeValue::Double),
        Some("float") => as_f64(v).map(|f| AttributeValue::Float(f as f32)),
        Some("int") => as_i64(v).map(|i| AttributeValue::Int(i as i32)),
        Some("uint") => as_i64(v).map(|i| AttributeValue::Uint(i as u32)),
        Some("short") => as_i64(v).map(|i| AttributeValue::Short(i as i16)),
        Some("ushort") => as_i64(v).map(|i| AttributeValue::Ushort(i as u16)),
        Some("int64") => as_i64(v).map(AttributeValue::Longlong),
        Some("uint64") => as_u64(v).map(AttributeValue::Ulonglong),
        Some("byte") => as_i64(v).map(|i| AttributeValue::Schar(i as i8)),
        Some("ubyte") => as_i64(v).map(|i| AttributeValue::Uchar(i as u8)),
        // Unknown / missing tag: widen.
        _ => match v {
            CborValue::Text(s) => Some(AttributeValue::Str(s.clone())),
            CborValue::Integer(_) => as_i64(v).map(AttributeValue::Longlong),
            CborValue::Float(f) => Some(AttributeValue::Double(*f)),
            _ => None,
        },
    }
}

/// Array form of [`cbor_to_attr_typed`]: dispatch to the plural `AttributeValue`
/// variant named by `tag`, or widen when the tag is unknown/absent.
fn cbor_array_to_attr(a: &[CborValue], tag: Option<&str>) -> Option<AttributeValue> {
    match tag {
        Some("string") => Some(AttributeValue::Strs(
            a.iter()
                .filter_map(|x| match x {
                    CborValue::Text(s) => Some(s.clone()),
                    _ => None,
                })
                .collect(),
        )),
        Some("double") => Some(AttributeValue::Doubles(
            a.iter().filter_map(as_f64).collect(),
        )),
        Some("float") => Some(AttributeValue::Floats(
            a.iter()
                .filter_map(|x| as_f64(x).map(|f| f as f32))
                .collect(),
        )),
        Some("int") => Some(AttributeValue::Ints(
            a.iter()
                .filter_map(|x| as_i64(x).map(|i| i as i32))
                .collect(),
        )),
        Some("uint") => Some(AttributeValue::Uints(
            a.iter()
                .filter_map(|x| as_i64(x).map(|i| i as u32))
                .collect(),
        )),
        Some("short") => Some(AttributeValue::Shorts(
            a.iter()
                .filter_map(|x| as_i64(x).map(|i| i as i16))
                .collect(),
        )),
        Some("ushort") => Some(AttributeValue::Ushorts(
            a.iter()
                .filter_map(|x| as_i64(x).map(|i| i as u16))
                .collect(),
        )),
        Some("int64") => Some(AttributeValue::Longlongs(
            a.iter().filter_map(as_i64).collect(),
        )),
        Some("uint64") => Some(AttributeValue::Ulonglongs(
            a.iter().filter_map(as_u64).collect(),
        )),
        Some("byte") => Some(AttributeValue::Schars(
            a.iter()
                .filter_map(|x| as_i64(x).map(|i| i as i8))
                .collect(),
        )),
        Some("ubyte") => Some(AttributeValue::Uchars(
            a.iter()
                .filter_map(|x| as_i64(x).map(|i| i as u8))
                .collect(),
        )),
        _ if a.iter().all(|x| matches!(x, CborValue::Text(_))) => Some(AttributeValue::Strs(
            a.iter()
                .filter_map(|x| match x {
                    CborValue::Text(s) => Some(s.clone()),
                    _ => None,
                })
                .collect(),
        )),
        _ if a.iter().all(|x| matches!(x, CborValue::Integer(_))) => Some(
            AttributeValue::Longlongs(a.iter().filter_map(as_i64).collect()),
        ),
        _ if a.iter().all(|x| matches!(x, CborValue::Float(_))) => Some(AttributeValue::Doubles(
            a.iter().filter_map(as_f64).collect(),
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{attr_value_to_cbor, attr_value_type_tag};

    /// Guard the two halves of the attribute-type vocabulary against drift: the
    /// tags are *produced* by [`crate::metadata::attr_value_type_tag`] (on
    /// import) and *consumed* by [`cbor_to_attr_typed`] (on export), in separate
    /// modules.  For every `AttributeValue` variant, tagging then reconstructing
    /// through the CBOR value must land on the same NetCDF type — otherwise a
    /// renamed or unhandled tag would silently widen (e.g. `float` → `double`).
    #[test]
    fn every_type_tag_round_trips_through_reconstruction() {
        use AttributeValue::*;
        // One scalar and one array sample per numeric/text family — covers all
        // 22 `AttributeValue` variants the producer can tag.
        let samples: &[AttributeValue] = &[
            Str("k".into()),
            Strs(vec!["a".into(), "b".into()]),
            Double(2.5),
            Doubles(vec![1.0, 2.0]),
            Float(1.5),
            Floats(vec![1.0, 2.0]),
            Int(-7),
            Ints(vec![1, 2]),
            Uint(9),
            Uints(vec![1, 2]),
            Short(3),
            Shorts(vec![1, 2]),
            Ushort(4),
            Ushorts(vec![1, 2]),
            Longlong(11),
            Longlongs(vec![1, 2]),
            Ulonglong(12),
            Ulonglongs(vec![1, 2]),
            Schar(-5),
            Schars(vec![1, 2]),
            Uchar(255),
            Uchars(vec![1, 2]),
        ];
        for original in samples {
            let tag = attr_value_type_tag(original);
            let cbor = attr_value_to_cbor(original);
            let restored = cbor_to_attr_typed(&cbor, Some(tag))
                .unwrap_or_else(|| panic!("tag {tag:?} reconstructed to no AttributeValue"));
            assert_eq!(
                attr_value_type_tag(&restored),
                tag,
                "tag {tag:?} reconstructed to a different NetCDF type: {restored:?}",
            );
        }
    }
}
