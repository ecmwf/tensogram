// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::collections::BTreeMap;
use std::path::Path;

use ciborium::Value as CborValue;
use eccodes::{CodesFile, FallibleIterator, KeyRead, ProductKind};

use tensogram_core::pipeline::apply_pipeline;
use tensogram_core::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata};
use tensogram_core::{encode, DataPipeline, Dtype, EncodeOptions};

use crate::error::GribError;
use crate::metadata::{extract_all_namespace_keys, extract_mars_keys, GribKeySet};

/// Options for GRIB → Tensogram conversion.
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// How to group GRIB messages into Tensogram messages.
    pub grouping: Grouping,
    /// Tensogram encode options (hash algorithm, etc.).
    pub encode_options: EncodeOptions,
    /// When `true`, extract keys from all ecCodes namespaces (ls, geography,
    /// time, vertical, parameter, statistics) and store them under a `"grib"`
    /// sub-object.  MARS keys always go in `"mars"` regardless of this flag.
    /// Default: `false`.
    pub preserve_all_keys: bool,
    /// Encoding/filter/compression pipeline for data objects.
    /// Defaults to all "none" (uncompressed raw float64).
    pub pipeline: DataPipeline,
}

/// How to group input GRIB messages.
#[derive(Debug, Clone)]
pub enum Grouping {
    /// Each GRIB message becomes one Tensogram message with one data object.
    OneToOne,
    /// All GRIB messages merged into a single Tensogram message
    /// with N data objects. Each `base[i]` holds ALL metadata
    /// for that object independently.
    MergeAll,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            grouping: Grouping::MergeAll,
            encode_options: EncodeOptions::default(),
            preserve_all_keys: false,
            pipeline: DataPipeline::default(),
        }
    }
}

/// Data extracted from a single GRIB message.
pub(crate) struct GribExtracted {
    pub(crate) keys: GribKeySet,
    pub(crate) values: Vec<f64>,
    pub(crate) shape: Vec<u64>,
    /// Non-mars namespace keys (`{ ns → { key → value } }`).
    /// `None` when `preserve_all_keys` is disabled.
    pub(crate) grib_keys: Option<BTreeMap<String, BTreeMap<String, CborValue>>>,
}

/// Convert all GRIB messages from a file path into Tensogram wire bytes.
///
/// Reads GRIB messages using ecCodes, extracts payload via `values` key
/// and MARS namespace keys (discovered dynamically), partitions keys into
/// common vs per-object, and produces one or more Tensogram messages.
///
/// MARS keys are stored under `base[i]["mars"]` for each data object.
/// The ecCodes `gridType` key is stored as `"grid"` within the mars namespace.
///
/// When `preserve_all_keys` is enabled, all non-mars namespace keys are
/// stored under `base[i]["grib"]` for each data object.
pub fn convert_grib_file(path: &Path, options: &ConvertOptions) -> Result<Vec<Vec<u8>>, GribError> {
    let mut handle = CodesFile::new_from_file(path, ProductKind::GRIB)?;

    let mut grib_messages = Vec::new();
    let mut iter = handle.ref_message_iter();
    while let Some(mut msg) = iter.next()? {
        let mut keys = extract_mars_keys(&mut msg)?;
        let values: Vec<f64> = msg.read_key("values")?;

        // gridType is outside the MARS namespace — read it separately
        // and store as "grid" within the mars key set.
        if let Ok(grid_type) = KeyRead::<String>::read_key(&msg, "gridType") {
            keys.keys
                .insert("grid".to_string(), CborValue::Text(grid_type));
        }

        // Optionally extract all non-mars namespace keys.
        let grib_keys = if options.preserve_all_keys {
            Some(extract_all_namespace_keys(&mut msg)?)
        } else {
            None
        };

        let ni: i64 = msg.read_key("Ni").unwrap_or(0);
        let nj: i64 = msg.read_key("Nj").unwrap_or(0);
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
            grib_keys,
        });
    }

    if grib_messages.is_empty() {
        return Err(GribError::NoMessages);
    }

    match options.grouping {
        Grouping::OneToOne => {
            convert_one_to_one(&grib_messages, &options.encode_options, &options.pipeline)
        }
        Grouping::MergeAll => {
            convert_merge_all(&grib_messages, &options.encode_options, &options.pipeline)
        }
    }
}

/// One GRIB → one Tensogram message (1:1 mapping).
///
/// All MARS keys go to `base[0]["mars"]`; all grib keys to `base[0]["grib"]`.
fn convert_one_to_one(
    messages: &[GribExtracted],
    encode_options: &EncodeOptions,
    pipeline: &DataPipeline,
) -> Result<Vec<Vec<u8>>, GribError> {
    let mut results = Vec::with_capacity(messages.len());

    for msg in messages {
        let mut entry = BTreeMap::new();
        if !msg.keys.keys.is_empty() {
            entry.insert("mars".to_string(), btree_to_cbor_map(&msg.keys.keys));
        }
        if let Some(grib) = &msg.grib_keys {
            if !grib.is_empty() {
                entry.insert("grib".to_string(), nested_btree_to_cbor_map(grib));
            }
        }

        let global_meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
            ..Default::default()
        };

        let (desc, data_bytes) = build_data_object(&msg.values, &msg.shape, pipeline)?;
        let encoded = encode(&global_meta, &[(&desc, &data_bytes)], encode_options)
            .map_err(|e| GribError::Encode(e.to_string()))?;

        results.push(encoded);
    }

    Ok(results)
}

/// All GRIBs → one Tensogram message with N data objects.
///
/// Each object gets ALL its metadata in `base[i]` independently.
/// No common/varying partitioning is needed — each base entry is self-contained.
fn convert_merge_all(
    messages: &[GribExtracted],
    encode_options: &EncodeOptions,
    pipeline: &DataPipeline,
) -> Result<Vec<Vec<u8>>, GribError> {
    // Build per-object base entries — each entry holds ALL metadata for that object.
    let base: Vec<BTreeMap<String, CborValue>> = messages
        .iter()
        .map(|msg| {
            let mut entry = BTreeMap::new();
            if !msg.keys.keys.is_empty() {
                entry.insert("mars".to_string(), btree_to_cbor_map(&msg.keys.keys));
            }
            if let Some(grib) = &msg.grib_keys {
                if !grib.is_empty() {
                    entry.insert("grib".to_string(), nested_btree_to_cbor_map(grib));
                }
            }
            entry
        })
        .collect();

    let global_meta = GlobalMetadata {
        version: 2,
        base,
        ..Default::default()
    };

    let mut descriptors_and_data = Vec::with_capacity(messages.len());
    for msg in messages {
        let (desc, data_bytes) = build_data_object(&msg.values, &msg.shape, pipeline)?;
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

// ── CBOR map helpers ─────────────────────────────────────────────────────────

/// Convert a flat `BTreeMap<String, CborValue>` into a `CborValue::Map`.
fn btree_to_cbor_map(map: &BTreeMap<String, CborValue>) -> CborValue {
    CborValue::Map(
        map.iter()
            .map(|(k, v)| (CborValue::Text(k.clone()), v.clone()))
            .collect(),
    )
}

/// Convert a nested `{ ns → { key → value } }` into a `CborValue::Map`
/// of maps — used for the `"grib"` section.
fn nested_btree_to_cbor_map(map: &BTreeMap<String, BTreeMap<String, CborValue>>) -> CborValue {
    CborValue::Map(
        map.iter()
            .map(|(ns, inner)| (CborValue::Text(ns.clone()), btree_to_cbor_map(inner)))
            .collect(),
    )
}

/// Build a `DataObjectDescriptor` + raw f64 bytes from GRIB values,
/// applying the configured encoding/filter/compression pipeline via the
/// shared [`tensogram_core::pipeline::apply_pipeline`] helper.
///
/// Byte order is little-endian: ecCodes returns native f64 values which
/// we serialize as LE bytes. MARS keys are carried in
/// `GlobalMetadata.base[i]["mars"]`, not in the descriptor params.
fn build_data_object(
    values: &[f64],
    shape: &[u64],
    pipeline: &DataPipeline,
) -> Result<(DataObjectDescriptor, Vec<u8>), GribError> {
    let ndim = shape.len() as u64;
    let mut strides = vec![0u64; shape.len()];
    if !shape.is_empty() {
        strides[shape.len() - 1] = 1;
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    let mut desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim,
        shape: shape.to_vec(),
        strides,
        dtype: Dtype::Float64,
        byte_order: ByteOrder::Little,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    };

    // GRIB data is always float64, so pass `Some(values)` — simple_packing
    // is always applicable if the user requests it.
    apply_pipeline(&mut desc, Some(values), pipeline, "GRIB message")
        .map_err(GribError::InvalidData)?;

    let data_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok((desc, data_bytes))
}
