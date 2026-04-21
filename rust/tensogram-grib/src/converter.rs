// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::path::Path;

use ciborium::Value as CborValue;
use eccodes::{CodesFile, FallibleIterator, KeyRead, ProductKind};

use tensogram::pipeline::apply_pipeline;
use tensogram::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata};
use tensogram::{DataPipeline, Dtype, EncodeOptions, encode};

use crate::error::GribError;
use crate::metadata::{GribKeySet, extract_all_namespace_keys, extract_mars_keys};

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
/// Reads GRIB messages using ecCodes, extracts each message's `values`
/// payload and its MARS namespace keys (discovered dynamically), and
/// produces one or more Tensogram messages depending on
/// [`ConvertOptions::grouping`].
///
/// MARS keys are stored under `base[i]["mars"]` for each data object —
/// every `base[i]` entry is self-contained (see `plans/DESIGN.md`, the
/// v0.6 metadata-major-refactor). The ecCodes `gridType` key is stored
/// as `"grid"` within the mars namespace.
///
/// When `preserve_all_keys` is enabled, all non-mars namespace keys
/// (`ls`, `geography`, `time`, `vertical`, `parameter`, `statistics`)
/// are also stored under `base[i]["grib"]` for each data object.
pub fn convert_grib_file(path: &Path, options: &ConvertOptions) -> Result<Vec<Vec<u8>>, GribError> {
    let mut handle = CodesFile::new_from_file(path, ProductKind::GRIB)?;
    let grib_messages = extract_messages(&mut handle, options.preserve_all_keys)?;
    finish_conversion(grib_messages, options)
}

/// Convert all GRIB messages from an in-memory buffer into Tensogram wire bytes.
///
/// Identical semantics to [`convert_grib_file`], but reads from memory
/// instead of a file path. Useful for byte-range downloads, in-memory
/// staging pipelines, and Python bindings where callers already hold the
/// GRIB bytes (e.g. from `requests.get(...).content` or a `BytesIO` buffer).
///
/// The `buffer` is consumed — ecCodes opens a `FILE*` over it via
/// `fmemopen(3)` and holds the allocation for the lifetime of the handle.
///
/// # Errors
///
/// - [`GribError::CodesError`] — `fmemopen` failed, the buffer is not
///   parseable GRIB, or an ecCodes call failed during message iteration.
/// - [`GribError::NoMessages`] — the buffer contained zero valid GRIB
///   messages (empty buffer, junk, or unrelated format).
/// - [`GribError::InvalidData`] — a GRIB grid dimension overflowed `u64`
///   (not reachable on 64-bit targets).
/// - [`GribError::Encode`] — the Tensogram encode stage rejected the
///   produced descriptors (e.g. bad pipeline arguments).
pub fn convert_grib_buffer(
    buffer: Vec<u8>,
    options: &ConvertOptions,
) -> Result<Vec<Vec<u8>>, GribError> {
    let mut handle = CodesFile::new_from_memory(buffer, ProductKind::GRIB)?;
    let grib_messages = extract_messages(&mut handle, options.preserve_all_keys)?;
    finish_conversion(grib_messages, options)
}

/// Extract every GRIB message from `handle` into our intermediate
/// [`GribExtracted`] form. Generic over the underlying storage type `D`
/// so we can share this loop between the file-path and in-memory entry
/// points (`CodesFile<File>` vs `CodesFile<Vec<u8>>`).
fn extract_messages<D: Debug>(
    handle: &mut CodesFile<D>,
    preserve_all_keys: bool,
) -> Result<Vec<GribExtracted>, GribError> {
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
        let grib_keys = if preserve_all_keys {
            Some(extract_all_namespace_keys(&mut msg)?)
        } else {
            None
        };

        let ni: i64 = msg.read_key("Ni").unwrap_or(0);
        let nj: i64 = msg.read_key("Nj").unwrap_or(0);
        let shape =
            if ni > 0 && nj > 0 {
                let nj = u64::try_from(nj)
                    .map_err(|_| GribError::InvalidData("Nj out of u64 range".into()))?;
                let ni = u64::try_from(ni)
                    .map_err(|_| GribError::InvalidData("Ni out of u64 range".into()))?;
                vec![nj, ni] // row-major: [lat, lon]
            } else {
                vec![u64::try_from(values.len()).map_err(|_| {
                    GribError::InvalidData("numberOfPoints out of u64 range".into())
                })?]
            };

        grib_messages.push(GribExtracted {
            keys,
            values,
            shape,
            grib_keys,
        });
    }
    Ok(grib_messages)
}

/// Finalise the conversion by checking we got at least one message and
/// dispatching to the appropriate grouping strategy.
fn finish_conversion(
    grib_messages: Vec<GribExtracted>,
    options: &ConvertOptions,
) -> Result<Vec<Vec<u8>>, GribError> {
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
        if let Some(grib) = &msg.grib_keys
            && !grib.is_empty()
        {
            entry.insert("grib".to_string(), nested_btree_to_cbor_map(grib));
        }

        let global_meta = GlobalMetadata {
            version: 3,
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
            if let Some(grib) = &msg.grib_keys
                && !grib.is_empty()
            {
                entry.insert("grib".to_string(), nested_btree_to_cbor_map(grib));
            }
            entry
        })
        .collect();

    let global_meta = GlobalMetadata {
        version: 3,
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
/// shared [`tensogram::pipeline::apply_pipeline`] helper.
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
        masks: None,
        hash: None,
    };

    // GRIB data is always float64, so pass `Some(values)` — simple_packing
    // is always applicable if the user requests it.
    apply_pipeline(&mut desc, Some(values), pipeline, "GRIB message")
        .map_err(GribError::InvalidData)?;

    let data_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok((desc, data_bytes))
}

#[cfg(test)]
mod tests {
    //! Tests for [`convert_grib_buffer`], including parity with
    //! [`convert_grib_file`].
    //!
    //! The two entry points share `extract_messages` + `finish_conversion`
    //! internally and diverge only at the `CodesFile` constructor
    //! (`new_from_file` vs `new_from_memory`).  `buffer_matches_file`
    //! locks in decoded-payload equality; the remaining tests exercise
    //! the buffer path in isolation across grouping modes, flags, and
    //! error cases.

    use std::path::PathBuf;

    use super::*;

    fn testdata(name: &str) -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("testdata");
        path.push(name);
        path
    }

    #[test]
    fn buffer_matches_file() {
        // Feed the same GRIB to both entry points and compare the
        // produced Tensogram bytes. `extract_messages` is generic over
        // `D: Debug`, so any divergence between the two code paths
        // would surface here.
        let path = testdata("2t.grib2");
        let bytes = std::fs::read(&path).expect("read 2t.grib2");

        let opts = ConvertOptions::default();
        let from_file = convert_grib_file(&path, &opts).expect("convert file");
        let from_buf = convert_grib_buffer(bytes, &opts).expect("convert buffer");

        // NOTE: hashes and CBOR provenance fields (time, uuid) can differ
        // between runs because `populate_reserved_provenance` stamps
        // epoch-seconds + a random uuid. We compare the decoded payloads
        // + descriptor shape/dtype/pipeline instead.
        assert_eq!(from_file.len(), from_buf.len());
        let opts = tensogram::DecodeOptions::default();
        for (a, b) in from_file.iter().zip(from_buf.iter()) {
            let (_, objs_a) = tensogram::decode(a, &opts).expect("decode file");
            let (_, objs_b) = tensogram::decode(b, &opts).expect("decode buffer");
            assert_eq!(objs_a.len(), objs_b.len());
            for ((da, ba), (db, bb)) in objs_a.iter().zip(objs_b.iter()) {
                assert_eq!(da.shape, db.shape);
                assert_eq!(da.dtype, db.dtype);
                assert_eq!(da.encoding, db.encoding);
                assert_eq!(da.filter, db.filter);
                assert_eq!(da.compression, db.compression);
                assert_eq!(ba, bb, "payload bytes must match");
            }
        }
    }

    #[test]
    fn buffer_rejects_garbage() {
        // A non-GRIB buffer must produce a clean error, not a panic.
        let garbage = b"this is not a grib message".to_vec();
        let result = convert_grib_buffer(garbage, &ConvertOptions::default());
        assert!(result.is_err(), "expected error on garbage input");
    }

    #[test]
    fn buffer_merge_all_grouping() {
        // Exercise the MergeAll path from memory — single output message
        // with one object, since 2t.grib2 has one GRIB message.
        let bytes = std::fs::read(testdata("2t.grib2")).expect("read 2t.grib2");
        let opts = ConvertOptions {
            grouping: Grouping::MergeAll,
            ..Default::default()
        };
        let result = convert_grib_buffer(bytes, &opts).expect("convert buffer");
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn buffer_one_to_one_grouping() {
        // Exercise the OneToOne path from memory. 2t.grib2 contains a
        // single GRIB message so we should still get one output.
        let bytes = std::fs::read(testdata("2t.grib2")).expect("read 2t.grib2");
        let opts = ConvertOptions {
            grouping: Grouping::OneToOne,
            ..Default::default()
        };
        let result = convert_grib_buffer(bytes, &opts).expect("convert buffer");
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn buffer_preserve_all_keys() {
        // Check that the `preserve_all_keys` flag routes through the
        // buffer path identically to the file path.
        let bytes = std::fs::read(testdata("2t.grib2")).expect("read 2t.grib2");
        let opts = ConvertOptions {
            preserve_all_keys: true,
            ..Default::default()
        };
        let messages = convert_grib_buffer(bytes, &opts).expect("convert buffer");
        let meta = tensogram::decode_metadata(&messages[0]).expect("decode metadata");
        assert!(
            meta.base.iter().any(|entry| entry.contains_key("grib")),
            "preserve_all_keys should populate the grib sub-object"
        );
    }

    #[test]
    fn buffer_empty_returns_error() {
        // An empty buffer must surface *some* error — not a panic and
        // not a silent success.  We intentionally do NOT assert on the
        // exact variant here: ecCodes' `fmemopen(3)` path on a zero-
        // length buffer raises `CodesError::LibcNonZero` on some
        // platforms before we ever get to the message iterator, so
        // `GribError::NoMessages` is not guaranteed.  What matters is
        // that the caller gets a clean `Result::Err`.
        let result = convert_grib_buffer(Vec::new(), &ConvertOptions::default());
        assert!(result.is_err(), "empty buffer must produce an error");
    }
}
