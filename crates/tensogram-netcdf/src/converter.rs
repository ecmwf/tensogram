use std::collections::BTreeMap;
use std::path::Path;

use ciborium::Value as CborValue;
use netcdf::types::{FloatType, IntType, NcVariableType};
use netcdf::AttributeValue;

use tensogram_core::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata};
use tensogram_core::{encode, Dtype};

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
    pub split_by: SplitBy,
    pub encode_options: tensogram_core::EncodeOptions,
    pub cf: bool,
}

/// Extracted data for one variable ready to encode.
struct ExtractedVar {
    #[allow(dead_code)]
    name: String,
    dtype: Dtype,
    shape: Vec<u64>,
    data_bytes: Vec<u8>,
    base_entry: BTreeMap<String, CborValue>,
}

/// Convert all variables from a NetCDF file into Tensogram wire bytes.
pub fn convert_netcdf_file(
    path: &Path,
    options: &ConvertOptions,
) -> Result<Vec<Vec<u8>>, NetcdfError> {
    let file = netcdf::open(path)?;

    let file_path_str = path.to_string_lossy().to_string();

    if let Ok(mut groups) = file.groups() {
        if groups.next().is_some() {
            eprintln!(
                "warning: {file_path_str}: sub-groups found; only root-group variables are converted"
            );
        }
    }

    let global_attrs = extract_global_attrs(&file);

    let mut extracted: Vec<ExtractedVar> = Vec::new();

    for var in file.variables() {
        let var_name = var.name();
        let vartype = var.vartype();

        match extract_variable(&var, &var_name, &vartype, options, &global_attrs) {
            Ok(Some(ev)) => extracted.push(ev),
            Ok(None) => {}
            Err(NetcdfError::UnsupportedType { name, reason }) => {
                eprintln!("warning: skipping variable '{name}': {reason}");
            }
            Err(e) => return Err(e),
        }
    }

    if extracted.is_empty() {
        return Err(NetcdfError::NoVariables);
    }

    match options.split_by {
        SplitBy::File => encode_as_one_message(&extracted, &options.encode_options),
        SplitBy::Variable => encode_one_per_variable(&extracted, &options.encode_options),
        SplitBy::Record => encode_by_record(path, options, &file_path_str),
    }
}

fn extract_variable(
    var: &netcdf::Variable<'_>,
    var_name: &str,
    vartype: &NcVariableType,
    options: &ConvertOptions,
    global_attrs: &BTreeMap<String, CborValue>,
) -> Result<Option<ExtractedVar>, NetcdfError> {
    match vartype {
        NcVariableType::Char | NcVariableType::String => {
            return Err(NetcdfError::UnsupportedType {
                name: var_name.to_string(),
                reason: format!("{vartype:?} variables are not supported"),
            });
        }
        NcVariableType::Compound(_)
        | NcVariableType::Opaque(_)
        | NcVariableType::Enum(_)
        | NcVariableType::Vlen(_) => {
            return Err(NetcdfError::UnsupportedType {
                name: var_name.to_string(),
                reason: format!("complex type {vartype:?} is not supported"),
            });
        }
        _ => {}
    }

    let dims = var.dimensions();
    let shape: Vec<u64> = dims.iter().map(|d| d.len() as u64).collect();

    let has_scale = var.attribute("scale_factor").is_some();
    let has_offset = var.attribute("add_offset").is_some();
    let needs_unpack = has_scale || has_offset;

    let (dtype, data_bytes) = if needs_unpack {
        read_and_unpack(var, var_name)?
    } else {
        read_native(var, var_name, vartype)?
    };

    let mut netcdf_meta = extract_var_attrs(var);

    if !global_attrs.is_empty() {
        netcdf_meta.insert(
            "_global".to_string(),
            CborValue::Map(
                global_attrs
                    .iter()
                    .map(|(k, v)| (CborValue::Text(k.clone()), v.clone()))
                    .collect(),
            ),
        );
    }

    let mut base_entry: BTreeMap<String, CborValue> = BTreeMap::new();
    base_entry.insert("name".to_string(), CborValue::Text(var_name.to_string()));

    if !netcdf_meta.is_empty() {
        base_entry.insert(
            "netcdf".to_string(),
            CborValue::Map(
                netcdf_meta
                    .iter()
                    .map(|(k, v)| (CborValue::Text(k.clone()), v.clone()))
                    .collect(),
            ),
        );
    }

    if options.cf {
        let cf_meta = extract_cf_attrs(var);
        if !cf_meta.is_empty() {
            base_entry.insert(
                "cf".to_string(),
                CborValue::Map(
                    cf_meta
                        .iter()
                        .map(|(k, v)| (CborValue::Text(k.clone()), v.clone()))
                        .collect(),
                ),
            );
        }
    }

    Ok(Some(ExtractedVar {
        name: var_name.to_string(),
        dtype,
        shape,
        data_bytes,
        base_entry,
    }))
}

fn read_native(
    var: &netcdf::Variable<'_>,
    var_name: &str,
    vartype: &NcVariableType,
) -> Result<(Dtype, Vec<u8>), NetcdfError> {
    match vartype {
        NcVariableType::Int(IntType::I8) => {
            let vals: Vec<i8> = var
                .get_values(..)
                .map_err(|e| NetcdfError::InvalidData(format!("reading '{var_name}': {e}")))?;
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok((Dtype::Int8, bytes))
        }
        NcVariableType::Int(IntType::U8) => {
            let vals: Vec<u8> = var
                .get_values(..)
                .map_err(|e| NetcdfError::InvalidData(format!("reading '{var_name}': {e}")))?;
            Ok((Dtype::Uint8, vals))
        }
        NcVariableType::Int(IntType::I16) => {
            let vals: Vec<i16> = var
                .get_values(..)
                .map_err(|e| NetcdfError::InvalidData(format!("reading '{var_name}': {e}")))?;
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok((Dtype::Int16, bytes))
        }
        NcVariableType::Int(IntType::U16) => {
            let vals: Vec<u16> = var
                .get_values(..)
                .map_err(|e| NetcdfError::InvalidData(format!("reading '{var_name}': {e}")))?;
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok((Dtype::Uint16, bytes))
        }
        NcVariableType::Int(IntType::I32) => {
            let vals: Vec<i32> = var
                .get_values(..)
                .map_err(|e| NetcdfError::InvalidData(format!("reading '{var_name}': {e}")))?;
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok((Dtype::Int32, bytes))
        }
        NcVariableType::Int(IntType::U32) => {
            let vals: Vec<u32> = var
                .get_values(..)
                .map_err(|e| NetcdfError::InvalidData(format!("reading '{var_name}': {e}")))?;
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok((Dtype::Uint32, bytes))
        }
        NcVariableType::Int(IntType::I64) => {
            let vals: Vec<i64> = var
                .get_values(..)
                .map_err(|e| NetcdfError::InvalidData(format!("reading '{var_name}': {e}")))?;
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok((Dtype::Int64, bytes))
        }
        NcVariableType::Int(IntType::U64) => {
            let vals: Vec<u64> = var
                .get_values(..)
                .map_err(|e| NetcdfError::InvalidData(format!("reading '{var_name}': {e}")))?;
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok((Dtype::Uint64, bytes))
        }
        NcVariableType::Float(FloatType::F32) => {
            let vals: Vec<f32> = var
                .get_values(..)
                .map_err(|e| NetcdfError::InvalidData(format!("reading '{var_name}': {e}")))?;
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok((Dtype::Float32, bytes))
        }
        NcVariableType::Float(FloatType::F64) => {
            let vals: Vec<f64> = var
                .get_values(..)
                .map_err(|e| NetcdfError::InvalidData(format!("reading '{var_name}': {e}")))?;
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok((Dtype::Float64, bytes))
        }
        _ => Err(NetcdfError::UnsupportedType {
            name: var_name.to_string(),
            reason: format!("unhandled type {vartype:?}"),
        }),
    }
}

fn read_and_unpack(
    var: &netcdf::Variable<'_>,
    var_name: &str,
) -> Result<(Dtype, Vec<u8>), NetcdfError> {
    let scale_factor = get_f64_attr(var, "scale_factor");
    let add_offset = get_f64_attr(var, "add_offset");
    let fill_value = get_f64_attr(var, "missing_value").or_else(|| get_fill_value_as_f64(var));

    let mut vals: Vec<f64> = var.get_values(..).map_err(|e| {
        NetcdfError::InvalidData(format!("reading '{var_name}' for unpacking: {e}"))
    })?;

    for v in &mut vals {
        if let Some(fv) = fill_value {
            if (*v - fv).abs() < f64::EPSILON * fv.abs().max(1.0) {
                *v = f64::NAN;
                continue;
            }
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

fn get_fill_value_as_f64(var: &netcdf::Variable<'_>) -> Option<f64> {
    get_f64_attr(var, "_FillValue")
}

fn extract_global_attrs(file: &netcdf::File) -> BTreeMap<String, CborValue> {
    let mut map = BTreeMap::new();
    for attr in file.attributes() {
        if let Ok(val) = attr.value() {
            if let Some(cbor) = attr_value_to_cbor(&val) {
                map.insert(attr.name().to_string(), cbor);
            }
        }
    }
    map
}

fn build_descriptor(ev: &ExtractedVar) -> DataObjectDescriptor {
    let ndim = ev.shape.len() as u64;
    let mut strides = vec![0u64; ev.shape.len()];
    if !ev.shape.is_empty() {
        strides[ev.shape.len() - 1] = 1;
        for i in (0..ev.shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * ev.shape[i + 1];
        }
    }
    DataObjectDescriptor {
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
    }
}

fn encode_as_one_message(
    extracted: &[ExtractedVar],
    encode_options: &tensogram_core::EncodeOptions,
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
        .map(|ev| (build_descriptor(ev), ev.data_bytes.as_slice()))
        .collect();

    let refs: Vec<(&DataObjectDescriptor, &[u8])> =
        descriptors_and_data.iter().map(|(d, b)| (d, *b)).collect();

    let encoded = encode(&global_meta, &refs, encode_options)
        .map_err(|e| NetcdfError::Encode(e.to_string()))?;

    Ok(vec![encoded])
}

fn encode_one_per_variable(
    extracted: &[ExtractedVar],
    encode_options: &tensogram_core::EncodeOptions,
) -> Result<Vec<Vec<u8>>, NetcdfError> {
    let mut results = Vec::with_capacity(extracted.len());

    for ev in extracted {
        let global_meta = GlobalMetadata {
            version: 2,
            base: vec![ev.base_entry.clone()],
            ..Default::default()
        };

        let desc = build_descriptor(ev);
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
                match extract_variable(&var, &var_name, &vartype, options, &global_attrs) {
                    Ok(Some(ev)) => extracted.push(ev),
                    Ok(None) => {}
                    Err(NetcdfError::UnsupportedType { name, reason }) => {
                        eprintln!("warning: skipping variable '{name}': {reason}");
                    }
                    Err(e) => return Err(e),
                }
                continue;
            }

            match extract_variable_record(
                &var,
                &var_name,
                &vartype,
                options,
                &global_attrs,
                record_idx,
                &unlimited_name,
            ) {
                Ok(Some(ev)) => extracted.push(ev),
                Ok(None) => {}
                Err(NetcdfError::UnsupportedType { name, reason }) => {
                    eprintln!("warning: skipping variable '{name}': {reason}");
                }
                Err(e) => return Err(e),
            }
        }

        if !extracted.is_empty() {
            let msgs = encode_as_one_message(&extracted, &options.encode_options)?;
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
) -> Result<Option<ExtractedVar>, NetcdfError> {
    match vartype {
        NcVariableType::Char | NcVariableType::String => {
            return Err(NetcdfError::UnsupportedType {
                name: var_name.to_string(),
                reason: format!("{vartype:?} variables are not supported"),
            });
        }
        NcVariableType::Compound(_)
        | NcVariableType::Opaque(_)
        | NcVariableType::Enum(_)
        | NcVariableType::Vlen(_) => {
            return Err(NetcdfError::UnsupportedType {
                name: var_name.to_string(),
                reason: format!("complex type {vartype:?} is not supported"),
            });
        }
        _ => {}
    }

    let dims = var.dimensions();
    let shape: Vec<u64> = dims
        .iter()
        .filter(|d| d.name() != unlimited_name)
        .map(|d| d.len() as u64)
        .collect();

    let unlimited_pos = dims
        .iter()
        .position(|d| d.name() == unlimited_name)
        .unwrap_or(0);

    let extents = build_record_extents(dims, unlimited_pos, record_idx);

    let (dtype, data_bytes) = read_native_extents(var, var_name, vartype, &extents)?;

    let mut netcdf_meta = extract_var_attrs(var);
    netcdf_meta.insert(
        "record_index".to_string(),
        CborValue::Integer((record_idx as i64).into()),
    );

    if !global_attrs.is_empty() {
        netcdf_meta.insert(
            "_global".to_string(),
            CborValue::Map(
                global_attrs
                    .iter()
                    .map(|(k, v)| (CborValue::Text(k.clone()), v.clone()))
                    .collect(),
            ),
        );
    }

    let mut base_entry: BTreeMap<String, CborValue> = BTreeMap::new();
    base_entry.insert("name".to_string(), CborValue::Text(var_name.to_string()));

    if !netcdf_meta.is_empty() {
        base_entry.insert(
            "netcdf".to_string(),
            CborValue::Map(
                netcdf_meta
                    .iter()
                    .map(|(k, v)| (CborValue::Text(k.clone()), v.clone()))
                    .collect(),
            ),
        );
    }

    if options.cf {
        let cf_meta = extract_cf_attrs(var);
        if !cf_meta.is_empty() {
            base_entry.insert(
                "cf".to_string(),
                CborValue::Map(
                    cf_meta
                        .iter()
                        .map(|(k, v)| (CborValue::Text(k.clone()), v.clone()))
                        .collect(),
                ),
            );
        }
    }

    Ok(Some(ExtractedVar {
        name: var_name.to_string(),
        dtype,
        shape,
        data_bytes,
        base_entry,
    }))
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
