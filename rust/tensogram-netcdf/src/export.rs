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
//! Milestone 1 scope: dimensions + variables (native dtype) + data + scalar
//! attributes, emitted as netCDF-4.  Exact attribute types, classic-format
//! matching, groups, chunking/compression, and CF re-packing are follow-ups.

use std::path::Path;

use ciborium::Value as CborValue;
use netcdf::AttributeValue;

use tensogram::types::{ByteOrder, GlobalMetadata};
use tensogram::{decode, DecodeOptions, Dtype};

use crate::error::NetcdfError;

/// Attribute keys used internally by the importer to carry structure, not real
/// NetCDF attributes — they must not be written back as attributes.
const INTERNAL_KEYS: &[&str] = &["_dims", "_file", "_global", "record_index"];

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

    // Global attributes (best-effort scalar/array types in milestone 1).
    for (name, value) in global_attrs(&meta) {
        if let Some(av) = cbor_to_attr(&value) {
            file.add_attribute(&name, av)?;
        }
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

/// Real variable attributes for object `i` (excludes internal `_*` keys).
fn var_attrs(meta: &GlobalMetadata, i: usize) -> Vec<(String, AttributeValue)> {
    let Some(nc) = netcdf_map(meta, i) else {
        return Vec::new();
    };
    nc.iter()
        .filter_map(|(k, v)| match k {
            CborValue::Text(name) if !INTERNAL_KEYS.contains(&name.as_str()) => {
                cbor_to_attr(v).map(|av| (name.clone(), av))
            }
            _ => None,
        })
        .collect()
}

/// Global attributes, read from `base[0].netcdf._global`.
fn global_attrs(meta: &GlobalMetadata) -> Vec<(String, CborValue)> {
    let Some(nc) = netcdf_map(meta, 0) else {
        return Vec::new();
    };
    match map_get(nc, "_global") {
        Some(CborValue::Map(g)) => g
            .iter()
            .filter_map(|(k, v)| match k {
                CborValue::Text(name) => Some((name.clone(), v.clone())),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Best-effort CBOR → NetCDF attribute value.  Milestone 1 widens integers to
/// `i64` and floats to `f64`; exact attribute-type preservation is a follow-up.
fn cbor_to_attr(v: &CborValue) -> Option<AttributeValue> {
    match v {
        CborValue::Text(s) => Some(AttributeValue::Str(s.clone())),
        CborValue::Integer(i) => i64::try_from(*i).ok().map(AttributeValue::Longlong),
        CborValue::Float(f) => Some(AttributeValue::Double(*f)),
        CborValue::Array(a) if a.iter().all(|x| matches!(x, CborValue::Text(_))) => {
            Some(AttributeValue::Strs(
                a.iter()
                    .filter_map(|x| match x {
                        CborValue::Text(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect(),
            ))
        }
        CborValue::Array(a) if a.iter().all(|x| matches!(x, CborValue::Integer(_))) => {
            let v: Option<Vec<i64>> = a
                .iter()
                .map(|x| match x {
                    CborValue::Integer(i) => i64::try_from(*i).ok(),
                    _ => None,
                })
                .collect();
            v.map(AttributeValue::Longlongs)
        }
        CborValue::Array(a) if a.iter().all(|x| matches!(x, CborValue::Float(_))) => {
            Some(AttributeValue::Doubles(
                a.iter()
                    .filter_map(|x| match x {
                        CborValue::Float(f) => Some(*f),
                        _ => None,
                    })
                    .collect(),
            ))
        }
        _ => None,
    }
}
