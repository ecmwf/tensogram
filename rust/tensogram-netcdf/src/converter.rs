// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::collections::BTreeMap;
use std::path::Path;

use ciborium::Value as CborValue;
use netcdf::AttributeValue;
use netcdf::types::{FloatType, IntType, NcVariableType};

use tensogram::pipeline::apply_pipeline;
use tensogram::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata};
use tensogram::{DataPipeline, Dtype, encode};

use crate::error::NetcdfError;
use crate::metadata::{attr_value_to_cbor, extract_cf_attrs, extract_var_attrs};

/// How to group NetCDF variables into Tensogram messages.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum SplitBy {
    /// All variables from one file → one Tensogram message with N objects.
    #[default]
    File,
    /// Each variable → its own Tensogram message.
    Variable,
    /// Split along the unlimited (record) dimension — one message per record.
    /// Errors if the file has no unlimited dimension.
    Record,
}

/// Options for NetCDF → Tensogram conversion.
#[derive(Debug, Clone, Default)]
pub struct ConvertOptions {
    /// How to group variables into Tensogram messages. See [`SplitBy`].
    pub split_by: SplitBy,
    /// Encode-side knobs passed verbatim to `tensogram::encode`.
    /// Most callers can leave this at `Default`.
    pub encode_options: tensogram::EncodeOptions,
    /// When `true`, lift the 16 allow-listed CF attributes into
    /// `base[i]["cf"]`. The full attribute dump is always available
    /// under `base[i]["netcdf"]` regardless.
    pub cf: bool,
    /// Encoding/filter/compression pipeline applied to every data object.
    /// Defaults to all `"none"` (raw little-endian payload, no compression).
    pub pipeline: DataPipeline,
}

/// Extracted data for one variable ready to encode.
struct ExtractedVar {
    name: String,
    dtype: Dtype,
    shape: Vec<u64>,
    data_bytes: Vec<u8>,
    base_entry: BTreeMap<String, CborValue>,
}

/// Wrap a `BTreeMap<String, CborValue>` into a `CborValue::Map` by
/// cloning every key/value into the CBOR key shape. This is the
/// one-liner for the "turn a Rust string map into a CBOR map" pattern
/// that used to appear inline at four call sites.
fn cbor_map_from(map: &BTreeMap<String, CborValue>) -> CborValue {
    CborValue::Map(
        map.iter()
            .map(|(k, v)| (CborValue::Text(k.clone()), v.clone()))
            .collect(),
    )
}

/// Build the `base[i]` entry for one variable: always contains `name`,
/// gains a `netcdf` sub-map when any attributes were extracted, and
/// gains a `cf` sub-map when CF extraction is enabled and at least one
/// allow-listed attribute is present.
fn build_base_entry(
    var_name: &str,
    netcdf_meta: &BTreeMap<String, CborValue>,
    cf_meta: Option<&BTreeMap<String, CborValue>>,
) -> BTreeMap<String, CborValue> {
    let mut base_entry: BTreeMap<String, CborValue> = BTreeMap::new();
    base_entry.insert("name".to_string(), CborValue::Text(var_name.to_string()));

    if !netcdf_meta.is_empty() {
        base_entry.insert("netcdf".to_string(), cbor_map_from(netcdf_meta));
    }

    if let Some(cf) = cf_meta
        && !cf.is_empty()
    {
        base_entry.insert("cf".to_string(), cbor_map_from(cf));
    }

    base_entry
}

/// Convert all variables from a NetCDF file into Tensogram wire bytes.
///
/// # Errors
///
/// - [`NetcdfError::Netcdf`] if the file cannot be opened or a read fails.
/// - [`NetcdfError::NoVariables`] if the root group has no supported
///   numeric variables (char/string/compound/vlen are silently skipped
///   with a warning — this error fires only when *every* variable is
///   unsupported).
/// - [`NetcdfError::NoUnlimitedDimension`] if `split_by = Record` is
///   requested on a file without an unlimited dimension.
/// - [`NetcdfError::InvalidData`] on unknown pipeline codec names,
///   `simple_packing` `compute_params` failures (rare — usually
///   all-NaN data, which is already filtered), or low-level read
///   errors in record-split mode.
/// - [`NetcdfError::Encode`] if the underlying Tensogram encoder rejects
///   the configured pipeline (e.g. `szip` on >32-bit samples).
pub fn convert_netcdf_file(
    path: &Path,
    options: &ConvertOptions,
) -> Result<Vec<Vec<u8>>, NetcdfError> {
    let file = netcdf::open(path)?;

    let file_path_str = path.to_string_lossy().to_string();

    if let Ok(mut groups) = file.groups()
        && groups.next().is_some()
    {
        eprintln!(
            "warning: {file_path_str}: sub-groups found; only root-group variables are converted"
        );
    }

    let global_attrs = extract_global_attrs(&file);

    let mut extracted: Vec<ExtractedVar> = Vec::new();

    for var in file.variables() {
        let var_name = var.name();
        let vartype = var.vartype();
        push_extracted_or_warn(
            extract_variable(&var, &var_name, &vartype, options, &global_attrs),
            &mut extracted,
        )?;
    }

    if extracted.is_empty() {
        return Err(NetcdfError::NoVariables);
    }

    match options.split_by {
        SplitBy::File => {
            encode_as_one_message(&extracted, &options.encode_options, &options.pipeline)
        }
        SplitBy::Variable => {
            encode_one_per_variable(&extracted, &options.encode_options, &options.pipeline)
        }
        SplitBy::Record => encode_by_record(path, options, &file_path_str),
    }
}

fn extract_variable(
    var: &netcdf::Variable<'_>,
    var_name: &str,
    vartype: &NcVariableType,
    options: &ConvertOptions,
    global_attrs: &BTreeMap<String, CborValue>,
) -> Result<ExtractedVar, NetcdfError> {
    reject_unsupported_vartype(var_name, vartype)?;

    let dims = var.dimensions();
    let shape: Vec<u64> = dims.iter().map(|d| d.len() as u64).collect();

    let has_scale = var.attribute("scale_factor").is_some();
    let has_offset = var.attribute("add_offset").is_some();
    let needs_unpack = has_scale || has_offset;

    let (dtype, data_bytes) = if needs_unpack {
        read_and_unpack(var, var_name)?
    } else {
        // Build full-range extents so we can funnel through the same
        // `read_native_extents` helper as the record-split path.
        let full_extents = build_full_extents(dims);
        read_native_extents(var, var_name, vartype, &full_extents)?
    };

    let mut netcdf_meta = extract_var_attrs(var);

    if !global_attrs.is_empty() {
        netcdf_meta.insert("_global".to_string(), cbor_map_from(global_attrs));
    }

    let cf_meta = options.cf.then(|| extract_cf_attrs(var));
    let base_entry = build_base_entry(var_name, &netcdf_meta, cf_meta.as_ref());

    Ok(ExtractedVar {
        name: var_name.to_string(),
        dtype,
        shape,
        data_bytes,
        base_entry,
    })
}

/// Build full-range `Extent::SliceCount` values — one per dimension —
/// that together cover every element of a variable. This is the
/// explicit equivalent of passing `..` to `netcdf::Variable::get_values`
/// and lets us funnel both the file-wide read and the record-split
/// read through a single [`read_native_extents`] implementation.
fn build_full_extents(dims: &[netcdf::Dimension<'_>]) -> Vec<netcdf::Extent> {
    dims.iter()
        .map(|d| netcdf::Extent::SliceCount {
            start: 0,
            stride: 1,
            count: d.len(),
        })
        .collect()
}

/// Reject NetCDF variable types that have no clean tensor representation.
///
/// `Char` and `String` carry text data that can't be flattened into a
/// numeric byte payload. `Compound`, `Opaque`, `Enum`, and `Vlen` are
/// NetCDF-4 user-defined types whose layout is implementation-defined
/// — supporting them is intentionally out of scope for v1.
///
/// Both `extract_variable` and `extract_variable_record` call this as
/// the first thing they do; the caller catches `UnsupportedType` and
/// downgrades it to a stderr warning + skip.
fn reject_unsupported_vartype(var_name: &str, vartype: &NcVariableType) -> Result<(), NetcdfError> {
    match vartype {
        NcVariableType::Char | NcVariableType::String => Err(NetcdfError::UnsupportedType {
            name: var_name.to_string(),
            reason: format!("{vartype:?} variables are not supported"),
        }),
        NcVariableType::Compound(_)
        | NcVariableType::Opaque(_)
        | NcVariableType::Enum(_)
        | NcVariableType::Vlen(_) => Err(NetcdfError::UnsupportedType {
            name: var_name.to_string(),
            reason: format!("complex type {vartype:?} is not supported"),
        }),
        _ => Ok(()),
    }
}

/// Append a freshly extracted variable to the working list, or downgrade
/// `UnsupportedType` errors to a stderr warning. Any other error type is
/// propagated to the caller. This dedupes the three nearly-identical
/// match blocks in `convert_netcdf_file` and `encode_by_record`.
fn push_extracted_or_warn(
    result: Result<ExtractedVar, NetcdfError>,
    extracted: &mut Vec<ExtractedVar>,
) -> Result<(), NetcdfError> {
    match result {
        Ok(ev) => {
            extracted.push(ev);
            Ok(())
        }
        Err(NetcdfError::UnsupportedType { name, reason }) => {
            eprintln!("warning: skipping variable '{name}': {reason}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Read all values as `f64`, apply CF-style unpacking
/// (`scale_factor` / `add_offset`), and replace fill values with NaN.
/// Output is always `Dtype::Float64` regardless of the on-disk dtype.
fn read_and_unpack(
    var: &netcdf::Variable<'_>,
    var_name: &str,
) -> Result<(Dtype, Vec<u8>), NetcdfError> {
    let scale_factor = get_f64_attr(var, "scale_factor");
    let add_offset = get_f64_attr(var, "add_offset");
    let fill_value = get_f64_attr(var, "missing_value").or_else(|| get_f64_attr(var, "_FillValue"));

    let mut vals: Vec<f64> = var.get_values(..).map_err(|e| {
        NetcdfError::InvalidData(format!("reading '{var_name}' for unpacking: {e}"))
    })?;

    for v in &mut vals {
        if let Some(fv) = fill_value
            && (*v - fv).abs() < f64::EPSILON * fv.abs().max(1.0)
        {
            *v = f64::NAN;
            continue;
        }
        if let Some(sf) = scale_factor {
            *v *= sf;
        }
        if let Some(ao) = add_offset {
            *v += ao;
        }
    }

    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok((Dtype::Float64, bytes))
}

/// Read a NetCDF variable attribute as an `f64`, coercing any numeric
/// scalar type. Returns `None` if the attribute is missing or non-numeric.
fn get_f64_attr(var: &netcdf::Variable<'_>, name: &str) -> Option<f64> {
    let attr = var.attribute(name)?;
    match attr.value().ok()? {
        AttributeValue::Double(v) => Some(v),
        AttributeValue::Float(v) => Some(v as f64),
        AttributeValue::Int(v) => Some(v as f64),
        AttributeValue::Short(v) => Some(v as f64),
        AttributeValue::Longlong(v) => Some(v as f64),
        _ => None,
    }
}

/// Extract every file-level (global) attribute as a CBOR map.
///
/// Mirrors [`metadata::extract_var_attrs`]: unreadable attributes
/// produce a stderr warning rather than silently disappearing.
fn extract_global_attrs(file: &netcdf::File) -> BTreeMap<String, CborValue> {
    let mut map = BTreeMap::new();
    for attr in file.attributes() {
        let name = attr.name();
        match attr.value() {
            Ok(val) => {
                map.insert(name.to_string(), attr_value_to_cbor(&val));
            }
            Err(e) => {
                eprintln!("warning: failed to read global attribute '{name}': {e}");
            }
        }
    }
    map
}

/// Build a `DataObjectDescriptor` from an extracted variable, applying the
/// requested encoding/filter/compression pipeline.
///
/// The raw `data_bytes` are passed to the Tensogram encoder unchanged — the
/// encoder runs the pipeline (simple_packing → shuffle → compression) when
/// the descriptor's `encoding`/`filter`/`compression` fields and `params`
/// map ask for it.
fn build_descriptor(
    ev: &ExtractedVar,
    pipeline: &DataPipeline,
) -> Result<DataObjectDescriptor, NetcdfError> {
    let ndim = ev.shape.len() as u64;
    let mut strides = vec![0u64; ev.shape.len()];
    if !ev.shape.is_empty() {
        strides[ev.shape.len() - 1] = 1;
        for i in (0..ev.shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * ev.shape[i + 1];
        }
    }
    let mut desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim,
        shape: ev.shape.clone(),
        strides,
        dtype: ev.dtype,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    };

    // Simple_packing is f64-only. Extract typed values from the raw
    // data_bytes ONLY when the variable is Float64 AND the caller asked
    // for simple_packing — otherwise skip the round-trip allocation.
    //
    // `chunks_exact(8)` guarantees 8-byte chunks; the `copy_from_slice`
    // cannot fail but we avoid `.unwrap()` anyway to keep the library
    // panic-free by construction.
    let f64_values: Option<Vec<f64>> =
        if ev.dtype == Dtype::Float64 && pipeline.encoding == "simple_packing" {
            Some(
                ev.data_bytes
                    .chunks_exact(8)
                    .map(|b| {
                        let mut buf = [0u8; 8];
                        buf.copy_from_slice(b);
                        f64::from_le_bytes(buf)
                    })
                    .collect(),
            )
        } else {
            None
        };

    // Delegate to the shared helper in tensogram::pipeline; map
    // its human-readable error string into NetcdfError::InvalidData.
    apply_pipeline(&mut desc, f64_values.as_deref(), pipeline, &ev.name)
        .map_err(NetcdfError::InvalidData)?;

    Ok(desc)
}

fn encode_as_one_message(
    extracted: &[ExtractedVar],
    encode_options: &tensogram::EncodeOptions,
    pipeline: &DataPipeline,
) -> Result<Vec<Vec<u8>>, NetcdfError> {
    let base: Vec<BTreeMap<String, CborValue>> =
        extracted.iter().map(|ev| ev.base_entry.clone()).collect();

    let global_meta = GlobalMetadata {
        version: 2,
        base,
        ..Default::default()
    };

    let descriptors_and_data: Vec<(DataObjectDescriptor, &[u8])> = extracted
        .iter()
        .map(|ev| Ok((build_descriptor(ev, pipeline)?, ev.data_bytes.as_slice())))
        .collect::<Result<Vec<_>, NetcdfError>>()?;

    let refs: Vec<(&DataObjectDescriptor, &[u8])> =
        descriptors_and_data.iter().map(|(d, b)| (d, *b)).collect();

    let encoded = encode(&global_meta, &refs, encode_options)
        .map_err(|e| NetcdfError::Encode(e.to_string()))?;

    Ok(vec![encoded])
}

fn encode_one_per_variable(
    extracted: &[ExtractedVar],
    encode_options: &tensogram::EncodeOptions,
    pipeline: &DataPipeline,
) -> Result<Vec<Vec<u8>>, NetcdfError> {
    let mut results = Vec::with_capacity(extracted.len());

    for ev in extracted {
        let global_meta = GlobalMetadata {
            version: 2,
            base: vec![ev.base_entry.clone()],
            ..Default::default()
        };

        let desc = build_descriptor(ev, pipeline)?;
        let encoded = encode(
            &global_meta,
            &[(&desc, ev.data_bytes.as_slice())],
            encode_options,
        )
        .map_err(|e| NetcdfError::Encode(e.to_string()))?;

        results.push(encoded);
    }

    Ok(results)
}

fn encode_by_record(
    path: &Path,
    options: &ConvertOptions,
    file_path_str: &str,
) -> Result<Vec<Vec<u8>>, NetcdfError> {
    let file = netcdf::open(path)?;

    let unlimited_dim = file
        .dimensions()
        .find(|d| d.is_unlimited())
        .map(|d| (d.name().to_string(), d.len()));

    let (unlimited_name, record_count) =
        unlimited_dim.ok_or_else(|| NetcdfError::NoUnlimitedDimension {
            file: file_path_str.to_string(),
        })?;

    if record_count == 0 {
        return Ok(vec![]);
    }

    let global_attrs = extract_global_attrs(&file);

    let mut results = Vec::with_capacity(record_count);

    for record_idx in 0..record_count {
        let mut extracted: Vec<ExtractedVar> = Vec::new();

        for var in file.variables() {
            let var_name = var.name();
            let vartype = var.vartype();

            let dims = var.dimensions();
            let has_unlimited = dims.iter().any(|d| d.name() == unlimited_name);

            if !has_unlimited {
                push_extracted_or_warn(
                    extract_variable(&var, &var_name, &vartype, options, &global_attrs),
                    &mut extracted,
                )?;
                continue;
            }

            push_extracted_or_warn(
                extract_variable_record(
                    &var,
                    &var_name,
                    &vartype,
                    options,
                    &global_attrs,
                    record_idx,
                    &unlimited_name,
                ),
                &mut extracted,
            )?;
        }

        if !extracted.is_empty() {
            let msgs =
                encode_as_one_message(&extracted, &options.encode_options, &options.pipeline)?;
            results.extend(msgs);
        }
    }

    Ok(results)
}

fn extract_variable_record(
    var: &netcdf::Variable<'_>,
    var_name: &str,
    vartype: &NcVariableType,
    options: &ConvertOptions,
    global_attrs: &BTreeMap<String, CborValue>,
    record_idx: usize,
    unlimited_name: &str,
) -> Result<ExtractedVar, NetcdfError> {
    reject_unsupported_vartype(var_name, vartype)?;

    let dims = var.dimensions();
    let shape: Vec<u64> = dims
        .iter()
        .filter(|d| d.name() != unlimited_name)
        .map(|d| d.len() as u64)
        .collect();

    // Invariant: encode_by_record only calls us with variables that
    // actually have the unlimited dimension in their dim list. If a
    // future caller violates that we'd rather get a clear error than
    // silently read from position 0.
    let unlimited_pos = dims
        .iter()
        .position(|d| d.name() == unlimited_name)
        .ok_or_else(|| {
            NetcdfError::InvalidData(format!(
                "extract_variable_record: variable '{var_name}' has no dimension \
                 matching unlimited '{unlimited_name}'"
            ))
        })?;

    let extents = build_record_extents(dims, unlimited_pos, record_idx);

    let (dtype, data_bytes) = read_native_extents(var, var_name, vartype, &extents)?;

    let mut netcdf_meta = extract_var_attrs(var);
    netcdf_meta.insert(
        "record_index".to_string(),
        CborValue::Integer((record_idx as i64).into()),
    );

    if !global_attrs.is_empty() {
        netcdf_meta.insert("_global".to_string(), cbor_map_from(global_attrs));
    }

    let cf_meta = options.cf.then(|| extract_cf_attrs(var));
    let base_entry = build_base_entry(var_name, &netcdf_meta, cf_meta.as_ref());

    Ok(ExtractedVar {
        name: var_name.to_string(),
        dtype,
        shape,
        data_bytes,
        base_entry,
    })
}

fn build_record_extents(
    dims: &[netcdf::Dimension<'_>],
    unlimited_pos: usize,
    record_idx: usize,
) -> Vec<netcdf::Extent> {
    dims.iter()
        .enumerate()
        .map(|(i, d)| {
            if i == unlimited_pos {
                netcdf::Extent::Index(record_idx)
            } else {
                netcdf::Extent::SliceCount {
                    start: 0,
                    stride: 1,
                    count: d.len(),
                }
            }
        })
        .collect()
}

fn read_native_extents(
    var: &netcdf::Variable<'_>,
    var_name: &str,
    vartype: &NcVariableType,
    extents: &[netcdf::Extent],
) -> Result<(Dtype, Vec<u8>), NetcdfError> {
    macro_rules! read_ext {
        ($t:ty, $dtype:expr) => {{
            let vals: Vec<$t> = var
                .get_values(extents)
                .map_err(|e| NetcdfError::InvalidData(format!("reading '{var_name}': {e}")))?;
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok(($dtype, bytes))
        }};
    }

    match vartype {
        NcVariableType::Int(IntType::I8) => read_ext!(i8, Dtype::Int8),
        NcVariableType::Int(IntType::U8) => {
            let vals: Vec<u8> = var
                .get_values(extents)
                .map_err(|e| NetcdfError::InvalidData(format!("reading '{var_name}': {e}")))?;
            Ok((Dtype::Uint8, vals))
        }
        NcVariableType::Int(IntType::I16) => read_ext!(i16, Dtype::Int16),
        NcVariableType::Int(IntType::U16) => read_ext!(u16, Dtype::Uint16),
        NcVariableType::Int(IntType::I32) => read_ext!(i32, Dtype::Int32),
        NcVariableType::Int(IntType::U32) => read_ext!(u32, Dtype::Uint32),
        NcVariableType::Int(IntType::I64) => read_ext!(i64, Dtype::Int64),
        NcVariableType::Int(IntType::U64) => read_ext!(u64, Dtype::Uint64),
        NcVariableType::Float(FloatType::F32) => read_ext!(f32, Dtype::Float32),
        NcVariableType::Float(FloatType::F64) => read_ext!(f64, Dtype::Float64),
        _ => Err(NetcdfError::UnsupportedType {
            name: var_name.to_string(),
            reason: format!("unhandled type {vartype:?}"),
        }),
    }
}
