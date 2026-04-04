use std::collections::BTreeMap;
use std::path::Path;

use ciborium::Value as CborValue;
use eccodes::{CodesFile, FallibleIterator, KeyRead, ProductKind};

use tensogram_core::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata};
use tensogram_core::{encode, Dtype, EncodeOptions};

use crate::error::GribError;
use crate::metadata::{extract_mars_keys, partition_keys, GribKeySet};

/// Options for GRIB → Tensogram conversion.
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// How to group GRIB messages into Tensogram messages.
    pub grouping: Grouping,
    /// Tensogram encode options (hash algorithm, etc.).
    pub encode_options: EncodeOptions,
}

/// How to group input GRIB messages.
#[derive(Debug, Clone)]
pub enum Grouping {
    /// Each GRIB message becomes one Tensogram message with one data object.
    OneToOne,
    /// All GRIB messages merged into a single Tensogram message
    /// with N data objects. Common MARS keys go to `common`,
    /// varying keys go to per-object descriptors.
    MergeAll,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            grouping: Grouping::MergeAll,
            encode_options: EncodeOptions::default(),
        }
    }
}

/// Data extracted from a single GRIB message.
pub(crate) struct GribExtracted {
    pub(crate) keys: GribKeySet,
    pub(crate) values: Vec<f64>,
    pub(crate) shape: Vec<u64>,
}

/// Convert all GRIB messages from a file path into Tensogram wire bytes.
///
/// Reads GRIB messages using ecCodes, extracts payload via `values` key
/// and MARS namespace keys, partitions keys into common vs per-object,
/// and produces one or more Tensogram messages.
pub fn convert_grib_file(
    path: &Path,
    options: &ConvertOptions,
) -> Result<Vec<Vec<u8>>, GribError> {
    let mut handle = CodesFile::new_from_file(path, ProductKind::GRIB)?;

    let mut grib_messages = Vec::new();
    let mut iter = handle.ref_message_iter();
    while let Some(msg) = iter.next()? {
        let keys = extract_mars_keys(&msg);
        let values: Vec<f64> = msg.read_key("values")?;

        let ni: i64 = msg.read_key("Ni").unwrap_or(0);
        let nj: i64 = msg.read_key("Nj").unwrap_or(0);
        // Regular grids have positive Ni/Nj; reduced grids (Ni=0) fall back to 1D.
        let shape = if ni > 0 && nj > 0 {
            let nj = u64::try_from(nj)
                .map_err(|_| GribError::InvalidData("Nj out of u64 range".into()))?;
            let ni = u64::try_from(ni)
                .map_err(|_| GribError::InvalidData("Ni out of u64 range".into()))?;
            vec![nj, ni] // row-major: [lat, lon]
        } else {
            vec![u64::try_from(values.len())
                .map_err(|_| GribError::InvalidData("numberOfPoints out of u64 range".into()))?]
        };

        grib_messages.push(GribExtracted {
            keys,
            values,
            shape,
        });
    }

    if grib_messages.is_empty() {
        return Err(GribError::NoMessages);
    }

    match options.grouping {
        Grouping::OneToOne => convert_one_to_one(&grib_messages, &options.encode_options),
        Grouping::MergeAll => convert_merge_all(&grib_messages, &options.encode_options),
    }
}

/// One GRIB → one Tensogram message (1:1 mapping).
fn convert_one_to_one(
    messages: &[GribExtracted],
    encode_options: &EncodeOptions,
) -> Result<Vec<Vec<u8>>, GribError> {
    let mut results = Vec::with_capacity(messages.len());

    for msg in messages {
        let global_meta = GlobalMetadata {
            version: 2,
            common: msg.keys.keys.clone(),
            ..Default::default()
        };

        let (desc, data_bytes) = build_data_object(&msg.values, &msg.shape, &BTreeMap::new());
        let encoded = encode(&global_meta, &[(&desc, &data_bytes)], encode_options)
            .map_err(|e| GribError::Encode(e.to_string()))?;

        results.push(encoded);
    }

    Ok(results)
}

/// All GRIBs → one Tensogram message with N data objects.
fn convert_merge_all(
    messages: &[GribExtracted],
    encode_options: &EncodeOptions,
) -> Result<Vec<Vec<u8>>, GribError> {
    let all_keys: Vec<&GribKeySet> = messages.iter().map(|m| &m.keys).collect();
    let (common, varying) = partition_keys(&all_keys);

    let global_meta = GlobalMetadata {
        version: 2,
        common,
        ..Default::default()
    };

    let mut descriptors_and_data = Vec::with_capacity(messages.len());
    for (msg, varying_keys) in messages.iter().zip(varying.iter()) {
        let (desc, data_bytes) = build_data_object(&msg.values, &msg.shape, varying_keys);
        descriptors_and_data.push((desc, data_bytes));
    }

    let refs: Vec<_> = descriptors_and_data
        .iter()
        .map(|(desc, data)| (desc, data.as_slice()))
        .collect();

    let encoded = encode(&global_meta, &refs, encode_options)
        .map_err(|e| GribError::Encode(e.to_string()))?;

    Ok(vec![encoded])
}

/// Build a DataObjectDescriptor + raw f64 bytes from GRIB values.
///
/// Byte order is little-endian: ecCodes returns native f64 values which we
/// serialize as LE bytes. The descriptor records this so decoders know.
fn build_data_object(
    values: &[f64],
    shape: &[u64],
    extra_params: &BTreeMap<String, CborValue>,
) -> (DataObjectDescriptor, Vec<u8>) {
    let ndim = shape.len() as u64;
    let mut strides = vec![0u64; shape.len()];
    if !shape.is_empty() {
        strides[shape.len() - 1] = 1;
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim,
        shape: shape.to_vec(),
        strides,
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: extra_params.clone(),
        hash: None,
    };

    let data_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    (desc, data_bytes)
}
